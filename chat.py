from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
# from azure.search.documents import SearchClient
# from azure.search.documents.models import SearchQuery
import os
from PIL import Image
from azure.ai.inference.prompts import PromptTemplate
from msrest.authentication import CognitiveServicesCredentials

project_connection_string = ""
endpoint = ""
key = ""
# search_endpoint = ""
# search_key = ""
# search_index_name = ""

project = AIProjectClient.from_connection_string(
    conn_str=project_connection_string, credential=DefaultAzureCredential()
)

chat = project.inference.get_chat_completions_client()
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)
# search_client = SearchClient(endpoint=search_endpoint, index_name=search_index_name, credential=search_key)

def upload_image(image):
    image_path = os.path.join("uploads", image.filename)
    with open(image_path, "wb") as f:
        f.write(image.read())
    return image_path

def analyze_image(image):
    visual_features =[
        VisualFeatures.TAGS,
        VisualFeatures.OBJECTS,
        VisualFeatures.READ,
        VisualFeatures.SMART_CROPS,
        VisualFeatures.CAPTION,
        VisualFeatures.DENSE_CAPTIONS
    ]
    result = client.analyze(
        image_data=image,
        visual_features=visual_features,
        smart_crops_aspect_ratios=[0.9, 1.33],
        gender_neutral_caption=True,
        language="en"
    )
    return result

# def search_incident_data(query):
#     search_query = SearchQuery(search_text=query)
#     results = search_client.search(search_query)
#     return results

def generate_incident_description(image_analysis):
    # Combine image analysis (and search results) to generate incident description Historical data: {search_results}
    description = f"Damage detected: {image_analysis.caption.text}."
    return description

def get_chat_response(message):
    system_message = {
        "role": "system",
        "content": """
            Based on the description below about an image, you need to create a detailed incident description for a 
            report with date and time details.

            Example:
            user: "A serious damage on the side of the container, probably the crane's lifting mechanism failed, causing the container 
            to drop from a height of about 10 feet"
            response: "On 5th October 2024, during routine loading operations at Port ABC, a crane malfunction
            occurred while lifting container number 267CY-04. The incident took place at approximately 
            02:15 PM. The crane's lifting mechanism failed, causing the container to drop from a height of 
            about 10 feet. The container sustained significant structural damage, including multiple dents 
            and a partially collapsed right front corner. Fortunately, no personnel were injured during the incident."
            """
    }
    
    user_message = {
        "role": "user",
        "content": message
    }
    
    messages = [system_message, user_message]
    
    response = chat.complete(
        model="gpt-4o",
        messages=messages,
        temperature=1,
        frequency_penalty=0.5,
        presence_penalty=0.5,
    )
    return response

def get_user_input(img: Image):
    return img

def get_image_from_folder(folder_path, image_name):
    # Load image to analyze into a 'bytes' object
    with open(folder_path + image_name, "rb") as f:
        image_data = f.read()
    return image_data

if __name__ == "__main__":
    """
    Main function to get user input and generate chat responses.
    """
    folder_path = "img/"
    image_name = "ABTest.png"
    image = get_image_from_folder(folder_path, image_name)
    # image_path = upload_image(image)
    image_analysis = analyze_image(image)
    # search_results = search_incident_data(image_analysis.description.captions[0].text)
    incident_description = generate_incident_description(image_analysis)
    response = get_chat_response(incident_description)
    print(response.choices[0].message.content)
