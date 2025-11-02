"""
Deepfake Detection Training Data Generation
Diverse prompts for creating fake face images
"""

# 인물 속성 변수
GENDERS = ["man", "woman", "person"]
AGES = ["child", "teenager", "young adult", "middle-aged", "elderly"]
ETHNICITIES = [
    # East Asian
    "Korean", "Japanese", "Chinese", "Taiwanese", "Vietnamese", "Thai", "Filipino",
    # South Asian
    "Indian", "Pakistani", "Bangladeshi", "Sri Lankan", "Nepalese",
    # Southeast Asian
    "Indonesian", "Malaysian", "Singaporean", "Burmese", "Cambodian",
    # Middle Eastern
    "Arab", "Turkish", "Persian", "Israeli", "Lebanese", "Egyptian",
    # European
    "British", "French", "German", "Italian", "Spanish", "Russian", "Polish", "Greek",
    "Swedish", "Norwegian", "Dutch", "Irish", "Portuguese", "Ukrainian",
    # African
    "Nigerian", "Ethiopian", "South African", "Kenyan", "Egyptian", "Moroccan",
    "Ghanaian", "Somali", "Sudanese", "Tanzanian",
    # Latin American
    "Mexican", "Brazilian", "Colombian", "Argentine", "Venezuelan", "Chilean",
    "Peruvian", "Cuban", "Dominican", "Puerto Rican",
    # North American
    "African American", "Native American", "Caucasian American",
    # Pacific
    "Australian", "New Zealander", "Polynesian", "Hawaiian",
    # Mixed
    "multiracial", "mixed ethnicity"
]
EXPRESSIONS = [
    "neutral expression",
    "smiling happily",
    "laughing",
    "serious expression",
    "surprised",
    "thoughtful",
    "confident",
    "gentle smile"
]

# 배경/환경 변수
ENVIRONMENTS = [
    "plain white background",
    "plain gray background",
    "plain black background",
    "soft gradient background",
    "blurred office background",
    "blurred outdoor park background",
    "blurred city street background",
    "professional studio lighting",
    "natural window light",
    "indoor home setting"
]

# 촬영 스타일
PHOTO_STYLES = [
    "professional headshot",
    "corporate portrait",
    "LinkedIn profile photo",
    "professional portrait photography",
    "studio portrait",
    "environmental portrait",
    "editorial portrait",
    "passport photo",
    "ID card photograph",
    "driver license photo",
    "university ID photo",
    "employee badge photo",
    "visa photograph",
    "professional actor headshot",
    "modeling portfolio photo",
    "business profile picture"
]

# 의상/외모
APPEARANCES = [
    "casual clothing",
    "business attire",
    "formal suit",
    "casual shirt",
    "hoodie",
    "t-shirt",
    "professional attire",
    "smart casual"
]

HAIR_STYLES = [
    "short hair",
    "long hair",
    "medium length hair",
    "curly hair",
    "straight hair",
    "ponytail",
    "bob cut",
    "buzz cut"
]

ACCESSORIES = [
    "wearing glasses",
    "no glasses",
    "wearing earrings",
    "wearing necklace",
    "no accessories",
    "minimal accessories",
    "wearing sunglasses",
    "wearing hat",
    "wearing scarf",
    "wearing watch"
]

# 피부톤 (더 세밀한 표현)
SKIN_TONES = [
    "fair skin",
    "light skin",
    "medium skin",
    "olive skin",
    "tan skin",
    "brown skin",
    "dark brown skin",
    "deep skin",
    "natural skin tone"
]

# 얼굴 특징
FACIAL_FEATURES = [
    "round face",
    "oval face",
    "square face",
    "heart-shaped face",
    "prominent cheekbones",
    "soft features",
    "strong jawline",
    "gentle features",
    "distinctive features"
]

# 품질 키워드 - Krea 모델에 최적화된 간결한 키워드
QUALITY_KEYWORDS = [
    "8K ultra detailed",
    "highly detailed facial features",
    "sharp focus",
    "professional photography",
    "studio lighting",
    "natural lighting",
    "photorealistic",
    "high resolution",
    "crisp details",
    "realistic skin texture"
]

def generate_prompt(
    gender=None,
    age=None,
    ethnicity=None,
    expression=None,
    environment=None,
    style=None,
    appearance=None,
    hair=None,
    accessory=None,
    skin_tone=None,
    facial_feature=None
):
    """
    Generate a single face image prompt with maximum diversity
    """
    import random

    # Random selection if not specified
    gender = gender or random.choice(GENDERS)
    age = age or random.choice(AGES)
    ethnicity = ethnicity or random.choice(ETHNICITIES)
    expression = expression or random.choice(EXPRESSIONS)
    environment = environment or random.choice(ENVIRONMENTS)
    style = style or random.choice(PHOTO_STYLES)
    appearance = appearance or random.choice(APPEARANCES)
    hair = hair or random.choice(HAIR_STYLES)
    accessory = accessory or random.choice(ACCESSORIES)
    skin_tone = skin_tone or random.choice(SKIN_TONES)
    facial_feature = facial_feature or random.choice(FACIAL_FEATURES)

    # Quality keywords - 2개만 선택 (더 간결하게)
    quality = ", ".join(random.sample(QUALITY_KEYWORDS, 2))

    # Reprompt 수준의 극도로 상세한 프롬프트 생성
    prompt = f"""A {style} captures a single {age} {ethnicity} {gender} positioned centrally in the frame, viewed from the shoulders up. The subject has {skin_tone} with {facial_feature}, and displays {expression}, their gaze directed straight at the camera with professional composure. Their {hair} frames their face naturally, and they are {accessory}. They wear {appearance}, which is rendered with realistic fabric textures and natural draping. The subject is positioned against {environment}, creating a clean, professional aesthetic. The scene is illuminated by {random.choice(['soft, diffused studio lighting from the front', 'natural window light from the side', 'professional three-point lighting setup', 'soft overhead lighting', 'balanced ambient studio light'])}, creating {random.choice(['subtle highlights on facial features', 'gentle shadows that define facial structure', 'even illumination across the face', 'soft contrast highlighting skin texture'])}. {quality}, photorealistic rendering with highly detailed facial features including skin pores, natural skin texture, realistic hair strands, and accurate fabric details. Sharp focus on the eyes and face, with {random.choice(['shallow depth of field', 'crisp focus throughout', 'slight background blur'])}. Single person only in frame, professional {style.lower()} composition."""

    return prompt

def generate_prompt_batch(num_prompts=100, balanced=True):
    """
    Generate a batch of diverse prompts

    Args:
        num_prompts: Number of prompts to generate
        balanced: If True, balance across demographics
    """
    import random

    prompts = []

    if balanced:
        # 균형잡힌 분포
        per_gender = num_prompts // len(GENDERS)

        for gender in GENDERS:
            for _ in range(per_gender):
                prompt = generate_prompt(gender=gender)
                prompts.append(prompt)
    else:
        # 완전 랜덤
        for _ in range(num_prompts):
            prompt = generate_prompt()
            prompts.append(prompt)

    # Shuffle
    random.shuffle(prompts)

    return prompts

# 프롬프트 카테고리별 생성 전략
def get_generation_strategy(total_images=2500):
    """
    Recommend generation strategy for given total
    """
    # 다양성을 위해 프롬프트 당 여러 seed 사용
    images_per_prompt = 5  # 같은 프롬프트, 다른 seed
    num_prompts = total_images // images_per_prompt

    strategy = {
        "total_images": total_images,
        "unique_prompts": num_prompts,
        "images_per_prompt": images_per_prompt,
        "estimated_time_hours": (total_images * 80) / 3600,  # 80초/이미지
        "demographics_balance": {
            "per_gender": num_prompts // len(GENDERS),
            "per_age": num_prompts // len(AGES),
            "per_ethnicity": num_prompts // len(ETHNICITIES)
        }
    }

    return strategy

# Example usage
if __name__ == "__main__":
    import json

    # 전략 출력
    for target in [1000, 2500, 5000]:
        print(f"\n=== Strategy for {target} images ===")
        strategy = get_generation_strategy(target)
        print(json.dumps(strategy, indent=2))

    # 샘플 프롬프트 생성
    print("\n=== Sample Prompts ===")
    samples = generate_prompt_batch(10)
    for i, prompt in enumerate(samples, 1):
        print(f"\n{i}. {prompt}")
