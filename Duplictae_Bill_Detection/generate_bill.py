from PIL import Image, ImageDraw, ImageFont
import os

# Function to generate a bill image
def generate_bill_image(description, amount, date, bill_number, output_path='bill_image.png'):
    # Define the image size and background color
    image_width, image_height = 600, 300
    background_color = (255, 255, 255)  # White background

    # Create a blank image with white background
    image = Image.new('RGB', (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    
    # Set font type and size
    font_path = os.path.join("arial.ttf")
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Set the starting position for the text
    text_x, text_y = 10, 10

    # Define the text content
    bill_content = f"""
    Description: {description}
    Amount: {amount}
    Date: {date}
    Bill Number: {bill_number}
    """

    # Add the text to the image
    draw.text((text_x, text_y), bill_content, font=font, fill=(0, 0, 0))  # Black text

    # Save the image
    image.save(output_path)
    print(f"Bill image saved to {output_path}")

# Example usage
description= "Office Supplies Notebooks"
amount = "93.65"
date = "2024-03-17"
bill_number = "B175360"

generate_bill_image(description, amount, date, bill_number, output_path='generated_bill_1.png')
