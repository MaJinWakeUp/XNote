from PIL import Image, ImageFilter

# Open the image
img = Image.open("example.png")

# Define the region to blur (left, upper, right, lower)
blur_box = (10, 5, 205, 60)  # Change as needed

# Crop the region, blur it, and paste it back
region = img.crop(blur_box)
blurred_region = region.filter(ImageFilter.GaussianBlur(radius=8))
img.paste(blurred_region, blur_box)

# Save the result
img.save("./example_blurred.png")