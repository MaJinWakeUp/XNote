from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.firefox import GeckoDriverManager
import time
import os
import requests

# Set Firefox options
firefox_options = Options()
# firefox_options.add_argument("--headless")  # Run without GUI
firefox_options.add_argument("--width=1920")
firefox_options.add_argument("--height=1080")

# Launch Firefox WebDriver
service = Service(GeckoDriverManager().install())
driver = webdriver.Firefox(service=service, options=firefox_options)

# Open the Perma.cc webpage
url = "https://perma.cc/75LH-32FH?view-mode=server-side"
driver.get(url)
time.sleep(10)  # Wait for JavaScript to load

# Scroll down to load dynamic content (if needed)
# driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
# time.sleep(2)

# Extract all text from the page
# xpath = "/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div[2]/div/div/div/div[1]/div[1]/div[1]/div[2]"
# element = driver.find_element(By.XPATH, xpath)
# element_text = element.text  # Get text content

text_elements = driver.find_elements(By.XPATH, "//p | //h1 | //h2 | //h3 | //span | //div")
text_content = "\n".join([el.text.strip() for el in text_elements if el.text.strip()])
print(text_content)

# Save extracted text
# with open("perma_text.txt", "w", encoding="utf-8") as file:
#     file.write(text_content)

# Extract image URLs
# image_elements = driver.find_elements(By.TAG_NAME, "img")
# os.makedirs("images", exist_ok=True)

# for index, img in enumerate(image_elements):
#     img_url = img.get_attribute("src")
#     if img_url:
#         if img_url.startswith("//"):
#             img_url = "https:" + img_url  # Fix relative URLs
        
#         # Download image
#         img_data = requests.get(img_url).content
#         with open(f"images/image_{index}.jpg", "wb") as img_file:
#             img_file.write(img_data)
#         print(f"Downloaded image_{index}.jpg")

# Close the browser
driver.quit()

print("Text and images extracted successfully!")
