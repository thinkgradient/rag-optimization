import os
from openai import AzureOpenAI
from pdf2image import convert_from_path
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
import base64
import json
import fitz

load_dotenv("variables.env")

# Load the environment variables
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
embedding_model_deployment = os.getenv("OPENAI_EMBEDDING_MODEL_DEPLOYMENT")
gpt_model_name = os.getenv("OPENAI_GPT_MODEL_NAME")
api_version = os.getenv("OPENAI_API_VERSION")

# Create an OpenAI assistant client
gpt_client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=api_base,
    api_version=api_version
)

def enhance_image(img):
    #enhance the image by converting it to grayscale, enhancing the contrast and the sharpness
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2)
    img = ImageEnhance.Sharpness(img).enhance(7.5)
    return img

def pdf_to_image(pdf_path, output_path="output_images", count=0):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Open the PDF
    doc = fitz.open(pdf_path)

    output_images_paths = []

    # Iterate over each page in the PDF
    for i, page in enumerate(doc):
        # Render the page to a Pixmap
        pix = page.get_pixmap()

        # Build an output path for the image
        image_path = os.path.join(output_path, f"page_{i + count + 1}.jpg")

        # Save the Pixmap as an image
        pix.save(image_path)
        print(f"Saved {os.path.basename(image_path)}")

        output_images_paths.append(image_path)

    return output_images_paths

def base64_encode_image(image_path):
    #encode image to base64
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def semantic_chunk(client=gpt_client, image_path="output_images/page_1.jpg", filename="test.pdf", model=gpt_model_name):
    #semantically chunk the text
    system_prompt_chunk = '''
        You will be provided with an image of a page from a PDF file, which contains information from a recruitment/job search context.

        Analyze the image provided and divide it into distinct chunks where each chunk of information can answer typical candidate or employer questions for a recruitment/job search scenario.  
        To chunk information effectively, follow the guidelines in order:  
        1. Group together any text found within the same structured element, such as a table, diagram, flowchart, or other graphic. Each structured element should be treated as a separate unit to preserve its specific context and meaning. Do not combine information from different structured elements into the same chunk.  
        ***It is extremely important that you recognize and capture any text within diagrams, graphics, flowcharts, or any other embedded image. These texts must be returned in your response.***  
        2. Any other text, such as free-standing paragraphs and sentences, should be divided and chunked based on semantic coherence. Combine paragraphs that discuss the same concept into one chunk. Separate paragraphs that cover different topics into distinct chunks to maintain clarity.  
        3. Ensure that free-standing titles and subtitles are included in the same chunk as the relevant content they introduce. This maintains the connection between headings and their associated information.

        The text will be in **English** as specified.

        Output as a Python array of chunks in JSON format with the following structure:
        {
            "chunks": [
                <chunk1>, 
                <chunk2>, 
                ...
            ]
        }
        Each chunk should be a JSON object with the following structure:
        {
            "title": <Title of the chunk as a String>,
            "content": <The unprocessed content of the chunk in an HTML format, as a String>,
            "keywords": <A List of keywords that describe the content of the chunk, where each keyword is a String>,
            "questions": <A List of potential questions that the content of the chunk can answer, where each question is a String>,
            "rephrased_summary": <A rephrased summary of the content of the chunk as a String>,
            "origin": <Name of the pdf file that will be passed to you alongside the image, as a String>,
            "explanation": <The reasoning that explains why you have decided to chunk the information in this way as a String. This is to be **in English**>
        }
        
        Example of a chunk with information relevant to a job posting:
        {
            "title": "Job Posting for Software Engineer",
            "content": "<table border='1' cellspacing='0' cellpadding='8'><tr><th>Field</th><th>Description</th></tr><tr><td>Job Title</td><td>Software Engineer</td></tr><tr><td>Location</td><td>Remote</td></tr><tr><td>Salary</td><td>$90,000 - $110,000 annually</td></tr><tr><td>Required Skills</td><td>Python, JavaScript, AWS, Docker</td></tr><tr><td>Experience Level</td><td>3-5 years</td></tr><tr><td>Benefits</td><td>Health insurance, 401(k), Paid time off</td></tr></table>",
            "keywords": ["Software Engineer", "Remote", "Salary", "Required Skills", "Benefits"],
            "questions": ["What is the salary for this software engineer role?", "What are the required skills for this job?", "Is the position remote or on-site?", "What benefits are offered?"],
            "rephrased_summary": "This job posting is for a Software Engineer role offering a salary of $90,000-$110,000 annually. It requires 3-5 years of experience and skills in Python, JavaScript, AWS, and Docker. Benefits include health insurance, 401(k), and paid time off.",
            "origin": "software_engineer_job_posting.pdf",
            "explanation": "This chunk contains a table that outlines key information related to the Software Engineer job posting, including salary, skills required, and benefits. The table format makes it a clear, self-contained unit of information."
        }
        If the image contains irrelevant information or no useful content, return the following:
        {
            "chunks": [
                {
                    "title": "Null",
                    "content": "N/A",
                    "keywords": [],
                    "questions": [],
                    "rephrased_summary": "No content or irrelevant information found"
                }
            ]
        }
    '''

    base64_image = base64_encode_image(image_path)
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages = [
            {"role": "system", "content": system_prompt_chunk},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}},
                    {"type": "text", "content": filename}
                ]
            }
        ],
        temperature=0.0,

    )

    try:
        return json.loads(response.choices[0].message.content)["chunks"]
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        # Print the first 100 characters of the response to check for obvious issues
        print("Response content:", response.choices[0].message.content)
        return None


def is_semantically_similar(chunk1, chunk2, client=gpt_client, model=gpt_model_name):

    #determine if two chunks are semantically similar
    system_prompt_similarity = '''
        Determine if the following two chunks of information are semantically similar or not, i.e., they talk about the **same** thing or **very similar** things.  
        The content of each chunk is in **English** and relates to **recruitment/job search** topics.

        If they are similar, provide a brief explanation of why you believe they are similar.  
        If they are not similar, provide a brief explanation of why you believe they are not similar.

        Output as a JSON object with the following structure:
        {
            "is_similar": <A boolean value indicating whether the two chunks are semantically similar or not - True or False>,
            "Explanation": <A string that details the reasoning behind why you think these chunks are semantically similar or different. This is to be **in English**>
        }
        If you come across any chunks in the following format:
        {
            "title": "Null",
            "content": "N/A",
            "keywords": [],
            "questions": [],
            "rephrased summary": "No content or irrelevant information found"
        }
        Automatically return the following:
        {
            "is_similar": False,
            "Explanation": "The chunks contain no content or irrelevant information."
        }
    '''

        

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages = [
            {"role": "system", "content": system_prompt_similarity},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(chunk1)},
                    {"type": "text", "text": json.dumps(chunk2)}
                ]
            }
        ],
        temperature=0.0,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        # Print the first 100 characters of the response to check for obvious issues
        print("Response content:", response.choices[0].message.content)
        return None


def semantic_cluster(chunks):
    #cluster the chunks based on semantic similarity
    #Used a naive approach to cluster the chunks - this can be improved by using a more sophisticated clustering algorithm, which can reduce number of API calls and improve efficiency 

    clusters = [] # List of dictionaries/JSON objects
    while chunks:
        first_chunk = chunks.pop(0)
        if first_chunk["title"] != "Null":
            current_cluster = [first_chunk]

            remaining_chunks = []
            for chunk in chunks:
                similarity = is_semantically_similar(first_chunk, chunk)
                if similarity is not None and similarity["is_similar"]:
                    current_cluster.append(chunk)
                    # print(f"""Chunks: {first_chunk["title"]} and {chunk["title"]}
                    #         Why are they supposedly similar?
                    #         {similarity["Explanation"]}
                    #         \n
                    # """) --> used for debugging purposes, uncomment to see the explanation for why two chunks are similar
                    try:
                        chunks.remove(chunk)
                    except ValueError:
                        pass
                else:
                    remaining_chunks.append(chunk)
            
            clusters.append({
                "chunks": current_cluster,
            })

            chunks = remaining_chunks
    return clusters


def chunk_and_cluster(output_paths):
    #chunk and cluster the images
    
    json_chunks = []
    cluster_path_text = r'.\clusters_text'
    cluster_path_json = r'.\clusters_json'
    
    #loop through the images and chunk them
    print("**Chunking in progress...**")
    for i, path in enumerate(output_paths):
        print(f"Chunking image {i+1}...")
        chunks = semantic_chunk(image_path=path[1], filename=path[0])
        if chunks is not None:
            json_chunks.extend(chunks)
            print(chunks)
            print("\n")
    print("**Chunking complete**\n\n\n")

    #cluster the chunks and save them to json files.
    print("**Clustering in progress...**")
    clusters = semantic_cluster(json_chunks)
    for i, cluster in enumerate(clusters):
        with open(os.path.join(cluster_path_text, f"Cluster{i+1}.txt"), "w") as file:
            json.dump(cluster["chunks"], file, indent=4)
            print(f"Cluster {i+1} saved")
        with open(os.path.join(cluster_path_json, f"Cluster{i+1}.json"), "w") as file:
            cluster_keywords = []
            cluster_questions = []
            for chunk in cluster["chunks"]:
                cluster_keywords.extend(chunk["keywords"])
                cluster_questions.extend(chunk["questions"])
            cluster_fields = {
                "keywords": cluster_keywords,
                "questions": cluster_questions
            }
            json.dump(cluster_fields, file, indent=4)
            print(f"Cluster {i+1} saved") 
        print("\n")
    print("**Clustering complete**")

def main():
    pdf_file_paths = ".\pdf_files"
    output_paths = []
    count = 0
    for pdf_file in os.listdir(pdf_file_paths):
        print(os.path.join(pdf_file_paths, pdf_file))
        page_paths = pdf_to_image(os.path.join(pdf_file_paths, pdf_file), count=count)
        for page_path in page_paths:
            pair = (pdf_file, page_path)
            output_paths.append(pair)
            count+=1
    #print(output_paths)
    chunk_and_cluster(output_paths)


if __name__ == "__main__":
    main()

