import os
import re

def find_math_blocks():
    math_indicators = ["+", "-", "*", "/", "gamma", "lambda", "alpha", "beta", "delta", "theta", "sum", "grad", "log", "max", "Q(", "V("]
    results = []
    
    for dirpath, _, filenames in os.walk("."):
        if "venv" in dirpath or ".git" in dirpath:
            continue
        for filename in filenames:
            if filename == "README.md":
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                code_blocks = re.findall(r"```(?!python|bash|sh|mermaid|html|css|javascript|js|yaml|yml)(.*?)```", content, re.DOTALL)
                for block in code_blocks:
                    block_clean = block.strip()
                    if any(ind in block_clean for ind in math_indicators) and len(block_clean) < 500:
                        results.append((filepath, block_clean))
                        
    # 寫入 UTF-8 檔案
    with open("math_blocks.txt", "w", encoding="utf-8") as out:
        for filepath, block in results:
            out.write(f"=== File: {filepath} ===\n")
            out.write(block)
            out.write("\n=======================\n\n")

if __name__ == "__main__":
    find_math_blocks()
