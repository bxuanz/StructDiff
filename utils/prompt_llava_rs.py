import torch
from PIL import Image


#  Description:
#  This script implements a three-stage prompt engineering pipeline. Key features include:
#  1. Fixed satellite viewpoint.
#  2. Strict final prompt length limit (under 70 words) to avoid CLIP token errors.
#  3. A critical rule to ensure exact instance counts from the user's original prompt
#     (e.g., "4 tennis courts") are preserved in the final output.



# -------------------------------------------------------------------------------
# STAGE 1: Geospatial Feature Interpretation
# -------------------------------------------------------------------------------
STAGE_1_REMOTE_SENSING_PROMPT = """
USER: <image>
You are an expert geospatial analyst. Your task is to interpret the provided structural map.
Do NOT describe the input's style (e.g., "line art," "boxes"). Instead, identify the real-world geospatial features these structures represent.
Provide your analysis as a simple list using this exact format. Do NOT include the viewpoint.
- Feature: [The type of geographic feature, e.g., River System, Urban Area, Aircraft]
- Structure: [A brief description of its shape or pattern, e.g., Meandering channel, Dense street grid, Irregular formation of 8 objects]

Provide the list only.
ASSISTANT:
"""

# -------------------------------------------------------------------------------
# STAGE 2: Semantic Alignment with User Intent
# -------------------------------------------------------------------------------
STAGE_2_REMOTE_SENSING_PROMPT = """
USER: You will be given a "Geospatial Analysis" list and a "User's Goal" sentence.
Your task is to refine the "Geospatial Analysis" list based on the user's goal.
1. For each "Feature" in the analysis, check if the "User's Goal" provides a more specific name for it.
2. If a more specific name is found, REPLACE the generic feature name with the specific one. Keep the "Structure" description.
3. Output ONLY the final, modified list.

---
EXAMPLE:
Geospatial Analysis:
- Feature: Multiple Sporting Courts
- Structure: Four rectangular courts arranged in a 2x2 grid.
User's Goal: "An aerial image containing 4 tennis courts"

Your output for this example should be:
- Feature: 4 tennis courts
- Structure: Four rectangular courts arranged in a 2x2 grid.
---

Now, perform this task with the following inputs:

Geospatial Analysis:
{stage_1_output}

User's Goal: "{raw_prompt}"
ASSISTANT:
"""


# # -------------------------------------------------------------------------------
# # STAGE 3: Professional Prompt Synthesis (with Instance Count Preservation)
# # -------------------------------------------------------------------------------
# STAGE_3_REMOTE_SENSING_PROMPT = """
# USER: You are an expert prompt engineer creating concise prompts for a satellite imagery model.
# Your task is to synthesize a "Refined Analysis" and an "Original Goal" into a single, professional prompt.

# **YOUR MOST IMPORTANT RULES:**
# 1.  **STARTING PHRASE**: The prompt **MUST** begin with the exact phrase "a satellite image of". No other words should come before it.
# 2.  **PRESERVE INSTANCE COUNTS**: Identify all specific object counts from the "Original Goal" (e.g., "4 tennis courts," "3 vehicles"). These counts are non-negotiable and **MUST** be included exactly as stated in the final prompt. This rule overrides all others if there is a conflict.

# Follow these instructions:
# 1.  **Theme**: After the starting phrase, describe the main theme from the "Original Goal".
# 2.  **Viewpoint**: The image MUST be from a **satellite remote sensing viewpoint** (e.g., nadir).
# 3.  **Details**: Weave in the key "Structure" details from the "Refined Analysis" to describe the layout.
# 4.  **Realism**: Use essential remote sensing keywords (e.g., high-resolution, multispectral, clear atmosphere).
# 5.  **Conciseness**: The final prompt must be a fluent paragraph and **strictly under 70 words**.

# CRITICAL: Do NOT use words like 'sketch', 'line art', 'canny', 'layout'. Describe a realistic, full-color satellite image.

# Provide ONLY the final, synthesized prompt.

# ---
# EXAMPLE:
# Refined Analysis:
# - Feature: 4 tennis courts
# - Structure: Four rectangular courts arranged in a 2x2 grid.
# - Feature: 3 vehicles
# - Structure: Three small, distinct rectangular shapes parked near a road.
# Original Goal: "An aerial image containing 4 tennis courts and 3 vehicles."

# Your output for this example should be:
# "a satellite image of a sports complex, high-resolution nadir view, featuring exactly 4 tennis courts in a 2x2 grid and 3 vehicles parked nearby. Photorealistic details, bright daylight with clear atmosphere, multispectral."
# ---

# Now, perform this task with the following inputs:

# Refined Analysis:
# {stage_2_output}

# Original Goal: "{raw_prompt}"
# ASSISTANT:
# """
# -------------------------------------------------------------------------------
# STAGE 3: Professional Prompt Synthesis (with Instance Count Preservation)
# -------------------------------------------------------------------------------
STAGE_3_REMOTE_SENSING_PROMPT = """
USER: You are an expert prompt engineer creating concise prompts for a satellite imagery model.
Your task is to synthesize a "Refined Analysis" and an "Original Goal" into a single, professional prompt.

**YOUR MOST IMPORTANT RULE:**
- **PRESERVE INSTANCE COUNTS**: Identify all specific object counts from the "Original Goal" (e.g., "4 tennis courts," "3 vehicles"). These counts are non-negotiable and **MUST** be included exactly as stated in the final prompt. This rule overrides all others if there is a conflict.

Follow these instructions:
1.  **Theme**: Use the "Original Goal" as the core theme, especially its object counts.
2.  **Viewpoint**: The image MUST be from a **satellite remote sensing viewpoint** (e.g., nadir).
3.  **Details**: Weave in the key "Structure" details from the "Refined Analysis" to describe the layout.
4.  **Realism**: Use essential remote sensing keywords (e.g., high-resolution, multispectral, clear atmosphere).
5.  **Conciseness**: The final prompt must be a fluent paragraph and **strictly under 70 words**.

CRITICAL: Do NOT use words like 'sketch', 'line art', 'canny', 'layout'. Describe a realistic, full-color satellite image.

Provide ONLY the final, synthesized prompt.

---
EXAMPLE:
Refined Analysis:
- Feature: 4 tennis courts
- Structure: Four rectangular courts arranged in a 2x2 grid.
- Feature: 3 vehicles
- Structure: Three small, distinct rectangular shapes parked near a road.
Original Goal: "An aerial image containing 4 tennis courts and 3 vehicles."

Your output for this example should be:
"High-resolution satellite image, nadir view of a sports complex. The scene must feature exactly 4 tennis courts in a 2x2 grid and 3 vehicles parked nearby. The layout is precise. Photorealistic, bright daylight, clear atmosphere, multispectral."
---

Now, perform this task with the following inputs:

Refined Analysis:
{stage_2_output}

Original Goal: "{raw_prompt}"
ASSISTANT:
"""


def _run_llava_generation(model, processor, text_prompt, image=None, device="cuda", dtype=torch.float16, max_new_tokens=200):
    if image:
        inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(device, dtype)
    else:
        inputs = processor(text=text_prompt, return_tensors="pt").to(device, dtype)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded_output = processor.decode(output[0], skip_special_tokens=True)
    if "ASSISTANT:" in decoded_output:
        assistant_response = decoded_output.split("ASSISTANT:")[-1].strip()
    else:
        assistant_response = decoded_output.strip()
    return assistant_response


def compose_prompt_llava_rs(model, processor, raw_prompt, image_path, device="cuda"):
    print("--- Starting Remote Sensing Prompt Engineering Pipeline ---")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return "Error: Input image file not found."
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print("\n[STAGE 1/3] Interpreting geospatial structures...")
    stage_1_output = _run_llava_generation(model, processor, STAGE_1_REMOTE_SENSING_PROMPT, image=image, device=device, dtype=torch_dtype)
    print(f"✅ Stage 1 Output (Geospatial Analysis):\n{stage_1_output}")

    print("\n[STAGE 2/3] Aligning analysis with user's goal...")
    stage_2_prompt_filled = STAGE_2_REMOTE_SENSING_PROMPT.format(stage_1_output=stage_1_output, raw_prompt=raw_prompt)
    stage_2_output = _run_llava_generation(model, processor, stage_2_prompt_filled, image=None, device=device, dtype=torch_dtype)
    print(f"✅ Stage 2 Output (Refined Analysis):\n{stage_2_output}")

    print("\n[STAGE 3/3] Synthesizing final professional prompt...")
    stage_3_prompt_filled = STAGE_3_REMOTE_SENSING_PROMPT.format(stage_2_output=stage_2_output, raw_prompt=raw_prompt)
    final_prompt = _run_llava_generation(model, processor, stage_3_prompt_filled, image=None, device=device, dtype=torch_dtype)
    print(f"\n✅ Final Generated Remote Sensing Prompt:\n{final_prompt}")

    print("\n--- Pipeline Finished Successfully ---")
    return final_prompt


# =================================================================================
#  EXAMPLE USAGE (with updated mock logic for the new requirement)
# =================================================================================
if __name__ == '__main__':
    class MockModel:
        def generate(self, **kwargs):
            text_input = processor.decode(kwargs['input_ids'][0])
            if "Geospatial Analysis" in text_input:
                if "User's Goal" in text_input: return [[0] * 6] # Stage 2
                else: return [[0] * 7] # Stage 3
            else: return [[0] * 4] # Stage 1

    class MockProcessor:
        def __call__(self, text, images=None, return_tensors=None):
            return {'input_ids': torch.tensor([list(text.encode())])}
        def decode(self, output_ids, skip_special_tokens=True):
            if len(output_ids) == 4: # Stage 1 Mock Output
                return "ASSISTANT: - Feature: Multiple Sporting Courts\n- Structure: Four rectangular courts arranged in a 2x2 grid.\n- Feature: Several Vehicles\n- Structure: Three small, distinct rectangular shapes parked near a road."
            elif len(output_ids) == 6: # Stage 2 Mock Output
                return "ASSISTANT: - Feature: 4 tennis courts\n- Structure: Four rectangular courts arranged in a 2x2 grid.\n- Feature: 3 vehicles\n- Structure: Three small, distinct rectangular shapes parked near a road."
            else: # Stage 3 Mock Output
                return """ASSISTANT: High-resolution satellite image, nadir view of a sports complex. The scene must feature exactly 4 tennis courts in a 2x2 grid and 3 vehicles parked nearby. The layout is precise. Photorealistic, bright daylight, clear atmosphere, multispectral."""

    print("=" * 60)
    print("RUNNING DEMO WITH MOCK MODEL AND PROCESSOR")
    print("=" * 60)
    mock_model = MockModel()
    processor = MockProcessor()
    
    # Using the user's new, more complex example prompt
    user_prompt_demo = "An aerial image containing 4 tennis courts and 3 vehicles."
    layout_image_path_demo = "dummy_image_path.png"
    Image.new('RGB', (100, 100), color='blue').save(layout_image_path_demo)

    final_generated_prompt = compose_prompt_llava_rs(
        model=mock_model,
        processor=processor,
        raw_prompt=user_prompt_demo,
        image_path=layout_image_path_demo,
        device="cpu"
    )

    print("\n--- FINAL PROMPT READY FOR TEXT-TO-IMAGE MODEL ---")
    print(final_generated_prompt)