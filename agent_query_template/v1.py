prompt = """
    Extract the key attributes that customers love about the product from the following review. 
    Provide a single-word or short-phrase attribute.
    
    Review: {review}
    
    Output:
"""

system_content = (
    "You are an AI extracting key product attributes from customer reviews."
)
