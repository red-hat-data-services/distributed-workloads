#!/usr/bin/env python3
"""
Synthetic Data Generation Job for Ray Cluster

This script runs on Ray workers to generate synthetic math problems
using lightweight local models loaded via transformers.

SHARED STORAGE INTEGRATION:
- Models and datasets cached to /shared/cache (shared PVC mount)
- Generated synthetic data saved to /shared/datasets
- Enables seamless data sharing between Ray jobs and PyTorchJobs
"""

import os
import json
import ray
import torch
import warnings
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


# Lightweight model configuration
LIGHTWEIGHT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@ray.remote(num_cpus=1, memory=4000)  # Optimized for TinyLlama
class LocalLightweightSDGWorker:
    def __init__(self, model_name: str = LIGHTWEIGHT_MODEL):
        """Initialize TinyLlama model for synthetic data generation"""
        self.model_name = model_name
        
        print(f"[Worker] Loading lightweight model: {self.model_name} (1.1B params)")
        print(f"[Worker] Model description: Good balance of speed/quality for math reasoning")
        
        # Determine shared cache directory - prioritize /shared for heavy data
        self.cache_dir = self._get_shared_cache_dir()
        print(f"[Worker] Using cache directory: {self.cache_dir}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with CPU/GPU compatibility
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True
        )
        
        # Move to appropriate device if not using device_map
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.device = device
        
        print(f"[Worker] Lightweight model loaded successfully on {self.device}")
    
    def _get_shared_cache_dir(self) -> str:
        """Get shared cache directory for heavy data (models, datasets)"""
        # Priority order: shared PVC -> workspace PVC -> local fallback
        possible_cache_paths = [
            "/shared/cache",           # Shared PVC mount (highest priority)
            "/tmp/.cache"             # Local fallback (ephemeral)
        ]
        
        for cache_path in possible_cache_paths:
            try:
                os.makedirs(cache_path, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(cache_path, ".cache_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                return cache_path
            except (OSError, PermissionError):
                continue
        
        # Fallback to /tmp if nothing else works
        return "/tmp/.cache"
    
    def generate_math_problems(self, gsm8k_samples: List[Dict]) -> List[Dict]:
        """Generate synthetic math problems from GSM8K samples"""
        all_results = []
        
        for i, seed_sample in enumerate(gsm8k_samples):
            try:
                # Generate 10 variations per seed with different difficulties
                difficulties = ["easy", "easy", "medium", "medium", "medium", "medium", "hard", "hard", "medium", "easy"]
                for var_id in range(10):
                    difficulty = difficulties[var_id]
                    prompt = self._create_variation_prompt(seed_sample, difficulty)
                    generated = self._generate_with_model(prompt)
                    
                    if generated and self._validate_math_problem(generated):
                        quality_scores = self._assess_quality(generated)
                        
                        all_results.append({
                            "question": generated["question"],
                            "answer": generated["answer"],
                            "context": seed_sample["question"],
                            "source": "granite_sdg_local",
                            "seed_id": seed_sample.get("seed_id", i),
                            "variation_id": var_id,
                            "difficulty": generated.get("difficulty", "medium"),
                            "concepts": generated.get("concepts", []),
                            "model_confidence": generated.get("confidence", 0.5),
                            "quality_scores": quality_scores,
                            "overall_quality": quality_scores["overall_quality"]
                        })
            except Exception as e:
                print(f"[Worker] Error processing seed {i}: {e}")
                continue
        
        return all_results
    
    def _create_variation_prompt(self, seed_sample: Dict, difficulty: str = "medium") -> str:
        return f"""You are an expert math teacher creating high-quality practice problems.

SEED PROBLEM:
Question: {seed_sample['question']}
Answer: {seed_sample['answer']}

TASK: Create a NEW math problem that follows these requirements:
1. Uses similar mathematical concepts but different context
2. Has different numbers and scenario
3. Difficulty level: {difficulty}
4. Must be solvable with clear step-by-step reasoning
5. Answer must be mathematically correct

OUTPUT FORMAT: Respond with ONLY valid JSON in this exact format:
{{
    "question": "A clear, well-formed math problem",
    "answer": "Step-by-step solution ending with final numerical answer",
    "difficulty": "{difficulty}",
    "concepts": ["list", "of", "math", "concepts"],
    "confidence": 0.95
}}

Generate the JSON now:"""
    
    def _generate_with_model(self, prompt: str) -> Dict:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return self._parse_response(response)
        except Exception as e:
            print(f"[Worker] Generation error: {e}")
            return None
    
    def _parse_response(self, response: str) -> Dict:
        """Parse JSON response with robust error handling"""
        try:
            # Clean the response - remove any text before/after JSON
            response = response.strip()
            
            # Find JSON boundaries
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                print(f"[Worker] No JSON found in response: {response[:100]}...")
                return self._fallback_parse(response)
            
            json_str = response[start_idx:end_idx]
            
            # Parse JSON
            import json
            parsed = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["question", "answer"]
            if not all(field in parsed for field in required_fields):
                print(f"[Worker] Missing required fields in JSON: {list(parsed.keys())}")
                return self._fallback_parse(response)
            
            # Add default values for optional fields
            parsed.setdefault("difficulty", "medium")
            parsed.setdefault("concepts", [])
            parsed.setdefault("confidence", 0.5)
            
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"[Worker] JSON decode error: {e}")
            return self._fallback_parse(response)
        except Exception as e:
            print(f"[Worker] Parsing error: {e}")
            return None
    
    def _fallback_parse(self, response: str) -> Dict:
        """Fallback parsing for non-JSON responses"""
        try:
            lines = response.strip().split('\n')
            # Try to extract question and answer from text
            question = ""
            answer = ""
            
            for i, line in enumerate(lines):
                line = line.strip()
                if any(keyword in line.lower() for keyword in ["question:", "problem:", "q:"]):
                    question = line.split(':', 1)[-1].strip()
                elif any(keyword in line.lower() for keyword in ["answer:", "solution:", "a:"]):
                    answer = '\n'.join(lines[i:]).split(':', 1)[-1].strip()
                    break
            
            if not question and lines:
                question = lines[0].strip()
            if not answer and len(lines) > 1:
                answer = '\n'.join(lines[1:]).strip()
            
            return {
                "question": question,
                "answer": answer,
                "difficulty": "unknown",
                "concepts": [],
                "confidence": 0.3  # Lower confidence for fallback parsing
            }
        except:
            return None
    
    def _validate_math_problem(self, generated: Dict) -> bool:
        if not generated or not generated.get("question") or not generated.get("answer"):
            return False
        
        question = generated["question"].lower()
        math_indicators = ["calculate", "solve", "find", "how many", "total", "cost"]
        has_math = any(indicator in question for indicator in math_indicators)
        
        return has_math and len(generated["question"]) >= 20 and len(generated["answer"]) >= 10
    
    def _assess_quality(self, generated: Dict) -> Dict:
        """Multi-dimensional quality assessment"""
        scores = {}
        
        # Basic validation
        question = generated.get("question", "").lower()
        answer = generated.get("answer", "").lower()
        
        # 1. Mathematical Content Assessment (0-1)
        math_indicators = ["calculate", "solve", "find", "how many", "total", "cost", "price", "sum", "difference"]
        math_operations = ["+", "-", "*", "/", "=", "×", "÷"]
        
        has_math_language = any(indicator in question for indicator in math_indicators)
        has_math_operations = any(op in answer for op in math_operations)
        has_numbers = any(char.isdigit() for char in (question + answer))
        
        scores["mathematical_content"] = (
            (0.4 if has_math_language else 0) +
            (0.3 if has_math_operations else 0) +
            (0.3 if has_numbers else 0)
        )
        
        # 2. Answer Quality Assessment (0-1)
        step_indicators = ["step", "first", "then", "next", "finally", "therefore"]
        has_reasoning = any(word in answer for word in step_indicators)
        has_final_answer = any(phrase in answer for phrase in ["answer is", "final answer", "result is", "equals"])
        answer_length_ok = 20 <= len(generated.get("answer", "")) <= 500
        
        scores["answer_quality"] = (
            (0.4 if has_reasoning else 0) +
            (0.3 if has_final_answer else 0) +
            (0.3 if answer_length_ok else 0)
        )
        
        # 3. Question Clarity Assessment (0-1)
        question_length_ok = 10 <= len(generated.get("question", "")) <= 200
        has_clear_ask = any(word in question for word in ["what", "how", "find", "calculate", "determine"])
        no_ambiguous_words = not any(word in question for word in ["maybe", "possibly", "might", "unclear"])
        
        scores["question_clarity"] = (
            (0.4 if question_length_ok else 0) +
            (0.3 if has_clear_ask else 0) +
            (0.3 if no_ambiguous_words else 0)
        )
        
        # 4. Confidence Score (from model if available)
        model_confidence = generated.get("confidence", 0.5)
        scores["model_confidence"] = min(max(model_confidence, 0), 1)
        
        # 5. Overall Quality Score (weighted average)
        weights = {
            "mathematical_content": 0.3,
            "answer_quality": 0.3,
            "question_clarity": 0.2,
            "model_confidence": 0.2
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights)
        scores["overall_quality"] = overall_score
        
        return scores


def _deduplicate_problems(problems: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
    """Remove near-duplicate problems based on question similarity"""
    if not problems:
        return problems
    
    # Simple deduplication based on question text similarity
    unique_problems = []
    seen_questions = []
    
    for problem in problems:
        question = problem.get("question", "").lower().strip()
        
        # Skip if too similar to existing questions
        is_duplicate = False
        for seen_q in seen_questions:
            # Simple similarity check - could be enhanced with more sophisticated methods
            similarity = _calculate_text_similarity(question, seen_q)
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_problems.append(problem)
            seen_questions.append(question)
    
    print(f"Deduplication: {len(problems)} -> {len(unique_problems)} problems")
    return unique_problems


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity based on word overlap"""
    if not text1 or not text2:
        return 0.0
    
    # Tokenize and normalize
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def _get_shared_cache_directory() -> str:
    """Get shared cache directory for heavy data (models, datasets)"""
    # Priority order: shared PVC -> workspace PVC -> local fallback
    possible_cache_paths = [
        "/shared/cache",           # Shared PVC mount (highest priority)
        "/tmp/.cache"             # Local fallback (ephemeral)
    ]
    
    for cache_path in possible_cache_paths:
        try:
            os.makedirs(cache_path, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(cache_path, ".cache_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return cache_path
        except (OSError, PermissionError):
            continue
    
    # Fallback to /tmp if nothing else works
    return "/tmp/.cache"


def main():
    """Main function for Ray job"""
    print("Starting distributed synthetic data generation...")
    
    # Set up shared cache directory for dataset loading
    shared_cache_dir = _get_shared_cache_directory()
    print(f"Using shared cache directory: {shared_cache_dir}")
    
    # Set environment variables for HuggingFace datasets to use shared cache
    os.environ['HF_HOME'] = shared_cache_dir
    os.environ['HF_DATASETS_CACHE'] = f"{shared_cache_dir}/datasets"
    os.environ['TRANSFORMERS_CACHE'] = f"{shared_cache_dir}/transformers"
    
    # Load GSM8K dataset to shared storage
    print("Loading GSM8K dataset to shared storage...")
    gsm8k_dataset = load_dataset("gsm8k", "main", cache_dir=f"{shared_cache_dir}/datasets")
    print(f"Dataset loaded: {len(gsm8k_dataset['train'])} train samples, {len(gsm8k_dataset['test'])} test samples")
    
    # Prepare seed samples
    num_seed_samples = 50
    seed_samples = []
    
    # Get the first num_seed_samples from the train split
    train_data = gsm8k_dataset["train"]
    
    # Validate first sample structure
    if len(train_data) > 0:
        first_sample = train_data[0]
        print(f"Sample structure: {type(first_sample)}, keys: {list(first_sample.keys()) if isinstance(first_sample, dict) else 'Not a dict'}")
    
    for i in range(min(num_seed_samples, len(train_data))):
        sample = train_data[i]
        seed_samples.append({
            "seed_id": i,
            "question": sample["question"],
            "answer": sample["answer"]
        })
    
    print(f"Using {len(seed_samples)} GSM8K problems as seeds")
    
    # Create Ray workers
    print(f"Using lightweight model: {LIGHTWEIGHT_MODEL} (1.1B params)")
    print(f"Model description: Good balance of speed/quality for math reasoning")
    
    num_workers = 3
    workers = [LocalLightweightSDGWorker.remote() for _ in range(num_workers)]
    print(f"Created {num_workers} Ray workers with 4GB memory each")
    
    # Distribute work
    samples_per_worker = len(seed_samples) // num_workers
    futures = []
    
    for i, worker in enumerate(workers):
        start_idx = i * samples_per_worker
        end_idx = start_idx + samples_per_worker if i < num_workers - 1 else len(seed_samples)
        worker_samples = seed_samples[start_idx:end_idx]
        
        future = worker.generate_math_problems.remote(worker_samples)
        futures.append(future)
    
    # Collect results
    all_problems = []
    for i, future in enumerate(futures):
        worker_results = ray.get(future)
        all_problems.extend(worker_results)
        print(f"Worker {i+1}: {len(worker_results)} problems")
    
    # Filter high-quality problems using multi-dimensional scoring
    quality_threshold = 0.7  # Slightly lower threshold for overall score
    min_mathematical_content = 0.6
    min_answer_quality = 0.5
    
    high_quality_problems = []
    for problem in all_problems:
        quality_scores = problem.get("quality_scores", {})
        overall_quality = quality_scores.get("overall_quality", 0)
        math_content = quality_scores.get("mathematical_content", 0)
        answer_quality = quality_scores.get("answer_quality", 0)
        
        # Multi-criteria filtering
        if (overall_quality >= quality_threshold and 
            math_content >= min_mathematical_content and 
            answer_quality >= min_answer_quality):
            high_quality_problems.append(problem)
    
    # Remove near-duplicates based on question similarity
    high_quality_problems = _deduplicate_problems(high_quality_problems)
    
    print(f"Total generated: {len(all_problems)}")
    print(f"High quality: {len(high_quality_problems)}")
    
    # Save dataset
    train_size = int(0.8 * len(high_quality_problems))
    # Calculate quality statistics
    if high_quality_problems:
        avg_overall_quality = sum(p["overall_quality"] for p in high_quality_problems) / len(high_quality_problems)
        difficulty_distribution = {}
        for p in high_quality_problems:
            diff = p.get("difficulty", "unknown")
            difficulty_distribution[diff] = difficulty_distribution.get(diff, 0) + 1
    else:
        avg_overall_quality = 0
        difficulty_distribution = {}
    
    synthetic_dataset = {
        "train": high_quality_problems[:train_size],
        "test": high_quality_problems[train_size:],
        "metadata": {
            "total_generated": len(all_problems),
            "high_quality_count": len(high_quality_problems),
            "quality_threshold": quality_threshold,
            "min_mathematical_content": min_mathematical_content,
            "min_answer_quality": min_answer_quality,
            "avg_overall_quality": round(avg_overall_quality, 3),
            "difficulty_distribution": difficulty_distribution,
            "model_used": LIGHTWEIGHT_MODEL,
            "generation_method": "lightweight_local_transformers",
            "features": [
                "structured_json_output",
                "multi_dimensional_quality_assessment", 
                "difficulty_variation",
                "deduplication",
                "robust_parsing"
            ]
        }
    }
    
    # Save to shared persistent storage
    # Priority order: shared PVC -> workspace PVC -> local fallback
    possible_paths = [
        "/shared/datasets",         # Shared PVC mount (highest priority)
        "/tmp/synthetic_data"      # Local fallback (ephemeral)
    ]
    
    output_path = None
    for path in possible_paths:
        try:
            os.makedirs(path, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(path, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            output_path = path
            print(f"Using storage path: {output_path}")
            break
        except (OSError, PermissionError):
            continue
    
    if not output_path:
        raise RuntimeError("No writable storage path found!")
    
    dataset_file = f"{output_path}/synthetic_dataset.json"
    with open(dataset_file, "w") as f:
        json.dump(synthetic_dataset, f, indent=2)
    
    print(f"Dataset saved: {len(synthetic_dataset['train'])} train / {len(synthetic_dataset['test'])} test")
    print(f"Saved to: {dataset_file}")
    
    # Also save metadata for debugging
    metadata_file = f"{output_path}/dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(synthetic_dataset["metadata"], f, indent=2)
    print(f"Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    main()
