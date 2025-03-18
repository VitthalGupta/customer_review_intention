prompt = """
    Extract the key attributes that customers love about the cosmetic product from the following review.  
    Provide **1-4 single-word attributes** that best describe the positive aspects.  
    Focus on common beauty-related or product satisfaction words such as:  
    **"fragrance", "texture", "hydration", "long lasting", "pigmentation", "moisturizing", "gentle", "shine", "non greasy", "lightweight", "allergy-free", "soothing", "effectiveness", "packaging", "climate suitability", "customer service", "quality", "overall satisfaction"**, etc.  

    Review: {review}  

    Output (comma-separated):
"""

system_content = "You are an AI review checker extracting key product attributes from customer reviews."
