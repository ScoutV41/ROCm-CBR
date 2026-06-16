import time
import platform
import subprocess
import json
import psutil
import gputil
import requests

def get_system_specs():
    """
    Gathers basic hardware information from the host machine.
    Automatically handles CPU, RAM, OS, and attempts to find GPUs.
    """
    specs = {
        "os": platform.system(),
        "os_release": platform.release(),
        "cpu_model": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu_model": "Unknown/CPU-Only",
        "gpu_vram_gb": 0.0,
        "gpu_driver": "Unknown"
    }

    # 1. Attempt to detect NVIDIA GPUs using GPUtil
    try:
        gpus = gputil.getGPUs()
        if gpus:
            primary_gpu = gpus[0]
            specs["gpu_model"] = primary_gpu.name
            specs["gpu_vram_gb"] = round(primary_gpu.memoryTotal / 1024, 2)
            specs["gpu_driver"] = primary_gpu.driver
            return specs  # Found NVIDIA, skip AMD check
    except Exception:
        # GPUtil might fail if no NVIDIA drivers are installed
        pass

    # 2. Attempt to detect AMD GPUs (fallback for Windows/Linux)
    if specs["os"] == "Windows":
        try:
            # Query WMI for graphics card details
            cmd = "wmic path win32_VideoController get name"
            output = subprocess.check_output(cmd, shell=True).decode().split('\n')
            # Filter out empty lines and headings
            gpu_names = [line.strip() for line in output if line.strip() and "Name" not in line]
            if gpu_names:
                # Basic check to look for AMD/Radeon strings
                amd_gpus = [g for g in gpu_names if "AMD" in g or "Radeon" in g]
                if amd_gpus:
                    specs["gpu_model"] = amd_gpus[0]
        except Exception:
            pass
    elif specs["os"] == "Linux":
        try:
            # Query lspci for VGA controllers matching AMD
            cmd = "lspci | grep -i 'vga\|3d' | grep -i 'AMD\|ATI'"
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            if output:
                specs["gpu_model"] = "AMD Linux GPU (Check rocm-smi)"
        except Exception:
            pass

    return specs

def run_inference_benchmark(model_name="qwen2.5:7b"):
    """
    Connects to a local Ollama instance, fires a standard prompt,
    and calculates tokens per second metrics.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": "Write a 300-word essay explaining quantum computing to a five-year-old.",
        "stream": False # Getting the full response at once makes parsing metrics simple
    }

    print(f"-> Starting benchmark against model: {model_name}...")
    print("-> Waiting for local inference engine response...")
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=120)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            
            # Ollama returns specific performance metrics in nanoseconds
            # eval_count = number of tokens in the prompt
            # eval_duration = time spent processing the prompt
            prompt_tokens = data.get("eval_count", 0)
            prompt_duration_sec = data.get("eval_duration", 1) / 1e9
            
            # predict_count = number of tokens generated in the answer
            # predict_duration = time spent generating the answer
            gen_tokens = data.get("predict_count", 0)
            gen_duration_sec = data.get("predict_duration", 1) / 1e9
            
            metrics = {
                "benchmark_status": "Success",
                "model_tested": model_name,
                "total_wall_time_sec": round(end_time - start_time, 2),
                "prompt_tokens": prompt_tokens,
                "generation_tokens": gen_tokens,
                "prompt_eval_tokens_per_sec": round(prompt_tokens / prompt_duration_sec, 2) if prompt_duration_sec > 0 else 0,
                "generation_tokens_per_sec": round(gen_tokens / gen_duration_sec, 2) if gen_duration_sec > 0 else 0
            }
            return metrics
        else:
            return {"benchmark_status": "Failed", "error": f"Ollama returned status code {response.status_code}"}
            
    except requests.exceptions.ConnectionError:
        return {"benchmark_status": "Failed", "error": "Could not connect to Ollama. Is it running?"}
    except Exception as e:
        return {"benchmark_status": "Failed", "error": str(e)}

def main():
    print("==============================================")
    print("   OPEN-SOURCE AI COMPUTE BENCHMARK RUNNER   ")
    print("==============================================\n")
    
    # Step 1: Gather System Specs
    print("[1/3] Profiling host hardware configuration...")
    hardware_profile = get_system_specs()
    print(f"    OS: {hardware_profile['os']} {hardware_profile['os_release']}")
    print(f"    CPU: {hardware_profile['cpu_model']} ({hardware_profile['cpu_cores_physical']} Cores)")
    print(f"    RAM: {hardware_profile['ram_gb']} GB")
    print(f"    GPU Detected: {hardware_profile['gpu_model']} (VRAM: {hardware_profile['gpu_vram_gb']} GB)")
    print("-" * 46)

    # Step 2: Run the actual AI Benchmark
    print("[2/3] Executing local inference test...")
    # Change "qwen2.5:7b" to whatever standard model you want your volunteers to use
    benchmark_results = run_inference_benchmark("qwen2.5:7b")
    
    if benchmark_results["benchmark_status"] == "Success":
        print(f"    Prompt Eval Speed: {benchmark_results['prompt_eval_tokens_per_sec']} t/s")
        print(f"    Generation Speed:  {benchmark_results['generation_tokens_per_sec']} t/s")
    else:
        print(f"    ERROR: {benchmark_results['error']}")
    print("-" * 46)

    # Step 3: Package everything together into your final JSON payload
    print("[3/3] Compiling final submission payload...")
    final_payload = {
        "timestamp": int(time.time()),
        "hardware": hardware_profile,
        "performance": benchmark_results
    }
    
    # Beautifully format the final JSON output
    print("\nFinal structured data ready to be uploaded to your database:")
    print(json.dumps(final_payload, indent=4))
    
    # TODO: Add your requests.post("https://your-api-endpoint.com/submit", json=final_payload) here!

if __name__ == "__main__":
    main()