import os
import uuid
import time
import requests
import base64
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from fasthtml.common import *


# Load environment variables
load_dotenv()

# In-memory storage
restoration_results = {}
analysis_cache = {}

# Restoration style options
RESTORATION_STYLES = [
    "Modern renovation", 
    "Historical restoration",
    "Eco-friendly renovation", 
    "Luxury upgrade",
    "Commercial conversion", 
    "Residential conversion",
    "Mixed-use development",
    "Minimalist restoration"
]

# Prompt template
RESTORATION_PROMPT = """
Create a realistic visualization of a derelict building after professional restoration and renovation.
{style_instruction}
{additional_instructions}
Maintain the same architectural footprint and core structure, but repair all damage.
Fix broken windows, repair the facade, update the exterior, and modernize the appearance while respecting the building's original character.
Make the surrounding area clean and well-maintained.
The result should look like a professional architectural visualization of the restored building.
"""

# Function to analyze building
def analyze_building_with_azure(image_data: str) -> str:
    try:
        image_hash = hash(image_data[:100])
        if image_hash in analysis_cache:
            print("✅ Using cached building analysis")
            return analysis_cache[image_hash]

        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")

        if not azure_api_key or not azure_endpoint:
            print("⚠️ Azure OpenAI credentials not found")
            return "Modern building with standard architectural features requiring restoration."

        print(f"🔍 Connecting to Azure AI Inference: {azure_endpoint} (deployment: {azure_deployment})")

        client = ChatCompletionsClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_api_key),
            deployment=azure_deployment  # Correct for azure-ai-inference v1.0.0b9
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this building image and describe its architectural style, condition, key features, and suggest specific restoration considerations. Focus on structural elements, materials, and historical significance if any. Keep response under 200 words."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ]

        response = client.complete(
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        analysis = response.choices[0].message.content
        analysis_cache[image_hash] = analysis

        print("✅ Azure AI Inference building analysis completed")
        return analysis

    except Exception as e:
        print(f"⚠️ Error with Azure AI Inference analysis: {e}")
        return "Modern building with standard architectural features requiring restoration."


# Function to generate restoration description
def generate_restoration_description_with_azure(prompt: str, building_analysis: str) -> str:
    try:
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")

        if not azure_api_key or not azure_endpoint:
            raise Exception("Azure credentials not available")

        print(f"📝 Generating detailed restoration description with GPT-4.1 at: {azure_endpoint} (deployment: {azure_deployment})")

        client = ChatCompletionsClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_api_key),
            deployment=azure_deployment
        )

        restoration_prompt = f"""
        Based on this building analysis: {building_analysis}

        And this restoration request: {prompt}

        Create a detailed, professional restoration plan that includes:
        1. Specific architectural improvements
        2. Materials and techniques to be used
        3. Timeline considerations
        4. Heritage preservation aspects
        5. Modern upgrades and sustainability features

        Write this as a comprehensive restoration proposal that could be presented to stakeholders.
        """

        messages = [{"role": "user", "content": restoration_prompt}]

        response = client.complete(
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )

        description = response.choices[0].message.content
        print("✅ Generated detailed restoration description")
        return description

    except Exception as e:
        print(f"⚠️ GPT-4.1 restoration description generation failed: {e}")
        raise e


# Function to create a restoration using Azure OpenAI image editing
def create_restoration_mockup(original_image_data: str, description: str) -> str:
    try:
        # Get Azure OpenAI credentials
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-image-1")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        
        if not endpoint or not api_key:
            print("⚠️ Azure OpenAI image credentials missing")
            return original_image_data
            
        # Construct the image editing URL
        if not endpoint.endswith("/"):
            endpoint += "/"
        url = f"{endpoint}openai/deployments/{deployment}/images/edits?api-version={api_version}"
        
        # Create restoration prompt based on the analysis
        prompt = f"""Restore this derelict building: clean the brickwork, repair windows, 
        add fresh paint, modern lighting, and surrounding greenery. Keep the original structure intact.
        
        Specific improvements: {description}"""
        
        # Decode base64 image data
        image_bytes = base64.b64decode(original_image_data)
        
        # Prepare the multipart form data
        files = {
            "image": ("building.png", image_bytes, "image/png")
        }
        data = {
            "prompt": prompt,
            "model": deployment,
            "size": "1024x1024",
            "quality": "high",
            "n": 1
        }
        headers = {
            "api-key": api_key
        }
        
        print("🎨 Sending image to Azure OpenAI for restoration...")
        response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result_data = response.json()
            if "data" in result_data and len(result_data["data"]) > 0:
                restored_b64 = result_data["data"][0]["b64_json"]
                print("✅ Image restoration completed successfully")
                return restored_b64
            else:
                print("⚠️ No image data in response")
                return original_image_data
        else:
            print(f"❌ Azure OpenAI image editing failed: {response.status_code}")
            if response.text:
                print(f"Error details: {response.text}")
            return original_image_data
            
    except Exception as e:
        print(f"⚠️ Error in create_restoration_mockup: {e}")
        return original_image_data


# Master function to orchestrate restoration
def restore_building_image(image_data: str, options: dict) -> dict:
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    print(f"🔑 Azure credentials available: {azure_api_key is not None and azure_endpoint is not None}")

    if not azure_api_key or not azure_endpoint:
        return {
            "error": "Azure credentials not found in environment variables.",
            "help": "Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and optionally AZURE_OPENAI_DEPLOYMENT_NAME environment variables."
        }

    result_id = uuid.uuid4().hex

    try:
        print("🔍 Analyzing building with Azure AI Inference GPT-4 Vision...")
        building_analysis = analyze_building_with_azure(image_data)

        selected_style = options.get("style", "Modern renovation")
        style_instruction = f"Use a {selected_style} style for the restoration."

        additional_instructions = [f"Building analysis: {building_analysis}"]

        if options.get("preserve_heritage", False):
            additional_instructions.append("Preserve historical and heritage elements of the building.")
        if options.get("landscaping", False):
            additional_instructions.append("Add attractive landscaping and greenery around the building.")
        if options.get("lighting", False):
            additional_instructions.append("Add modern and attractive lighting to highlight architectural features.")
        if options.get("expand_building", False):
            additional_instructions.append("Consider a tasteful expansion or addition that complements the original structure.")

        additional_instructions_text = " ".join(additional_instructions)

        prompt = RESTORATION_PROMPT.format(
            style_instruction=style_instruction,
            additional_instructions=additional_instructions_text
        )

        print("📝 Generating detailed restoration plan with GPT-4.1...")
        try:
            restoration_description = generate_restoration_description_with_azure(prompt, building_analysis)
        except Exception as e:
            print(f"⚠️ GPT-4.1 description generation failed: {e}")
            restoration_description = f"Restoration plan for {selected_style} style renovation based on the analysis."

        print("📸 Creating AI-powered restoration...")
        try:
            restored_img_data = create_restoration_mockup(image_data, restoration_description)
            restoration_success = restored_img_data != image_data  # Check if restoration actually happened
        except Exception as e:
            print(f"⚠️ Restoration failed: {e}")
            restored_img_data = image_data
            restoration_success = False

        result_data = {
            "id": result_id,
            "style": selected_style,
            "prompt": prompt,
            "original_image": image_data,
            "restored_image": restored_img_data,
            "options": options,
            "azure_analysis": building_analysis,
            "restoration_description": restoration_description,
            "restoration_success": restoration_success,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        restoration_results[result_id] = result_data
        print(f"✅ Restoration result stored in memory with ID: {result_id}")

        return result_data

    except Exception as e:
        return {
            "error": f"Restoration planning failed: {str(e)}",
            "help": "Please check your Azure AI credentials and GPT-4.1 deployment",
            "id": result_id
        }


# Set up the FastHTML app with required headers
app, rt = fast_app(
    hdrs=(
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@3.9.2/dist/full.css"),
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
        Script(src="https://unpkg.com/htmx.org@1.9.10"),
        # Add custom theme styles
        Style("""
            :root {
            --color-base-100: oklch(98% 0.002 247.839);
            --color-base-200: oklch(96% 0.003 264.542);
            --color-base-300: oklch(92% 0.006 264.531);
            --color-base-content: oklch(21% 0.034 264.665);
            --color-primary: oklch(47% 0.196 209.957);
            --color-primary-content: oklch(97% 0.014 254.604);
            --color-secondary: oklch(74% 0.134 119.635);
            --color-secondary-content: oklch(13% 0.028 261.692);
            --color-accent: oklch(71% 0.134 41.252);
            --color-accent-content: oklch(97% 0.014 254.604);
            --color-neutral: oklch(13% 0.028 261.692);
            --color-neutral-content: oklch(98% 0.002 247.839);
            --color-info: oklch(58% 0.158 241.966);
            --color-info-content: oklch(97% 0.013 236.62);
            --color-success: oklch(62% 0.194 149.214);
            --color-success-content: oklch(98% 0.018 155.826);
            --color-warning: oklch(66% 0.179 58.318);
            --color-warning-content: oklch(98% 0.022 95.277);
            --color-error: oklch(59% 0.249 0.584);
            --color-error-content: oklch(97% 0.014 343.198);
            }

            .text-arch-blue { color: oklch(47% 0.196 209.957); }
            .bg-renew-green { background-color: oklch(74% 0.134 119.635); }
            .custom-border { border-color: var(--color-base-300); }

            .comparison-slider {
                position: relative;
                width: 100%;
                overflow: hidden;
                border-radius: 0.5rem;
                margin: 1rem 0;
            }
            
            .before-after-container {
                position: relative;
                width: 100%;
                height: 400px;
            }
            
            .before-image, .after-image {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            
            .after-container {
                position: absolute;
                top: 0;
                left: 0;
                width: 50%;
                height: 100%;
                overflow: hidden;
            }
            
            .slider-handle {
                position: absolute;
                top: 0;
                bottom: 0;
                left: 50%;
                width: 4px;
                background: white;
                transform: translateX(-50%);
                cursor: ew-resize;
                z-index: 10;
            }
            
            .slider-handle::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 30px;
                height: 30px;
                background: white;
                border-radius: 50%;
                box-shadow: 0 0 5px rgba(0,0,0,0.5);
            }
            
            .slider-label {
                position: absolute;
                top: 10px;
                padding: 5px 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                border-radius: 4px;
                font-size: 12px;
                z-index: 5;
            }
            
            .before-label { left: 10px; }
            .after-label { right: 10px; }
        """),
    )
)

# Homepage Route - Building Restoration Dashboard
@rt("/")
def homepage():
    """Render the building restoration dashboard"""
    
    # Check if Azure credentials are available
    azure_available = bool(os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"))
    
    # Create toggle switches for restoration options
    def create_toggle(name, label, checked=False):
        return Div(
            Label(
                Input(
                    type="checkbox",
                    name=name,
                    checked="checked" if checked else None,
                    cls="toggle toggle-primary mr-3"
                ),
                Span(label),
                cls="label cursor-pointer justify-start"
            ),
            cls="mb-3"
        )
    
    # Create style selection dropdown
    def create_style_dropdown():
        options = []
        for style in RESTORATION_STYLES:
            options.append(Option(style, value=style))
            
        return Div(
            Label("Restoration Style", cls="label font-medium mb-2"),
            Select(
                *options,
                name="style",
                cls="select select-bordered w-full"
            ),
            cls="mb-4"
        )
    
    # API status alert
    api_status_alert = ""
    if not azure_available:
        api_status_alert = Div(
            Div(
                "⚠️ Azure OpenAI credentials missing - Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT",
                cls="alert alert-warning text-sm mb-4"
            )
        )
    
    # Restoration options panel
    restoration_options = Div(
        H3("Restoration Options", cls="text-lg font-semibold mb-4 text-arch-blue"),
        create_style_dropdown(),
        create_toggle("preserve_heritage", "Preserve Heritage Elements"),
        create_toggle("landscaping", "Add Landscaping & Greenery"),
        create_toggle("lighting", "Enhance with Architectural Lighting"),
        create_toggle("expand_building", "Consider Tasteful Expansion"),
        cls="mb-6 p-4 bg-base-200 rounded-lg"
    )
    
    # Building image upload section
    upload_section = Div(
        Label("Upload Building Image", cls="block text-xl font-medium mb-2 text-arch-blue"),
        P("Upload an image of a derelict building to visualize its restoration.", cls="mb-4"),
        Div(
            Label(
                Div(
                    Span("Click or drag image here", cls="text-lg text-center"),
                    P("Select a building photo (JPEG, PNG)", cls="text-sm text-center mt-2"),
                    cls="flex flex-col items-center justify-center h-full"
                ),
                Input(
                    type="file",
                    name="building_image",
                    accept="image/jpeg,image/png",
                    cls="hidden",
                    id="image-input"
                ),
                cls="w-full h-40 border-2 border-dashed rounded-lg flex items-center justify-center cursor-pointer hover:bg-base-200 transition-colors"
            ),
            cls="mb-6"
        ),
        
        # Preview area
        Div(
            Img(
                id="image-preview",
                src="",
                cls="max-h-64 mx-auto hidden object-contain rounded-lg border shadow-sm"
            ),
            cls="mb-6"
        ),
        cls="mb-8"
    )
    
    # Control panel 
    control_panel = Div(
        H2("Building Restoration Visualizer", cls="text-xl font-bold mb-4 text-arch-blue"),
        P("✨ Enhanced with Azure OpenAI Image Editing + GPT-4 Analysis", cls="text-sm text-secondary mb-4"),
        api_status_alert,
        upload_section,
        restoration_options,
        Button(
            "Generate Restoration",
            cls="btn btn-primary w-full",
            id="restore-button",
            disabled="disabled"
        ),
        cls="w-full md:w-1/2 bg-base-100 p-6 rounded-lg shadow-lg custom-border border"
    )
    
    # Results panel
    results_panel = Div(
        H2("Restoration Results", cls="text-xl font-bold mb-4 text-arch-blue"),
        Div(
            Div(
                cls="loading loading-spinner loading-lg text-primary",
                id="loading-indicator"
            ),
            cls="flex justify-center items-center h-32 hidden"
        ),
        Div(
            P("Upload a building image and click 'Generate Restoration' to see results.", 
              cls="text-center text-base-content/70 italic"),
            id="results-placeholder",
            cls="text-center py-12"
        ),
        
        # Container for results
        Div(
            # Before/After comparison slider
            Div(
                id="comparison-container",
                cls="hidden"
            ),
            
            # Details about the restoration
            Div(
                id="restoration-details",
                cls="mt-6 hidden"
            ),
            
            id="results-content",
            cls="hidden"
        ),
        
        # Actions for results
        Div(
            Button(
                "Download Restored Image",
                cls="btn btn-outline btn-accent btn-sm mr-2",
                id="download-button"
            ),
            Button(
                "New Restoration",
                cls="btn btn-outline btn-primary btn-sm",
                id="new-button"
            ),
            cls="mt-6 flex justify-end items-center gap-2 hidden",
            id="result-actions"
        ),
        cls="w-full md:w-1/2 bg-base-100 p-6 rounded-lg shadow-lg custom-border border"
    )
    
    # Add script for form handling
    form_script = Script("""
    document.addEventListener('DOMContentLoaded', function() {
        // Form elements
        const imageInput = document.getElementById('image-input');
        const imagePreview = document.getElementById('image-preview');
        const restoreButton = document.getElementById('restore-button');
        
        // Results elements
        const loadingIndicator = document.getElementById('loading-indicator').parentElement;
        const resultsPlaceholder = document.getElementById('results-placeholder');
        const resultsContent = document.getElementById('results-content');
        const comparisonContainer = document.getElementById('comparison-container');
        const restorationDetails = document.getElementById('restoration-details');
        const resultActions = document.getElementById('result-actions');
        const downloadButton = document.getElementById('download-button');
        const newButton = document.getElementById('new-button');
        
        // State variables
        let originalImageData = null;
        let restoredImageData = null;
        
        // Get options from the form
        function getOptions() {
            return {
                style: document.querySelector('select[name="style"]').value,
                preserve_heritage: document.querySelector('input[name="preserve_heritage"]').checked,
                landscaping: document.querySelector('input[name="landscaping"]').checked,
                lighting: document.querySelector('input[name="lighting"]').checked,
                expand_building: document.querySelector('input[name="expand_building"]').checked
            };
        }
        
        // Handle image upload
        imageInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            
            if (!file) {
                resetForm();
                return;
            }
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
                restoreButton.disabled = false;
                
                // Store the base64 data (remove the data URL prefix)
                originalImageData = e.target.result.split(',')[1];
            };
            
            reader.readAsDataURL(file);
        });
        
        // Reset the form
        function resetForm() {
            imageInput.value = '';
            imagePreview.src = '';
            imagePreview.classList.add('hidden');
            restoreButton.disabled = true;
            originalImageData = null;
        }
        
        // Handle restore button click
        restoreButton.addEventListener('click', function() {
            // Show loading state
            loadingIndicator.classList.remove('hidden');
            resultsPlaceholder.classList.add('hidden');
            resultsContent.classList.add('hidden');
            resultActions.classList.add('hidden');
            restoreButton.disabled = true;
            
            // Send request to API
            fetch('/restore', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image_data: originalImageData,
                    options: getOptions()
                })
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                loadingIndicator.classList.add('hidden');
                
                if (data.error) {
                    // Show error message with help text if available
                    let errorMessage = `Error: ${data.error}`;
                    
                    if (data.help) {
                        errorMessage += `<div class="mt-2 text-sm">${data.help}</div>`;
                        errorMessage += `<div class="mt-3 p-3 bg-base-300 rounded text-sm">
                            <strong>Troubleshooting:</strong>
                            <ul class="list-disc list-inside mt-1">
                                <li>Check your Azure OpenAI credentials</li>
                                <li>Verify your Azure OpenAI deployment supports image editing</li>
                                <li>Verify your accounts have proper billing setup</li>
                            </ul>
                        </div>`;
                    }
                    
                    comparisonContainer.innerHTML = `
                        <div class="alert alert-error">
                            <span>${errorMessage}</span>
                        </div>
                    `;
                    comparisonContainer.classList.remove('hidden');
                    resultsContent.classList.remove('hidden');
                    restoreButton.disabled = false;
                    return;
                }
                
                // Store the restored image data
                restoredImageData = data.restored_image;
                
                // Create the before/after comparison slider
                createComparisonSlider(originalImageData, restoredImageData);
                
                // Create restoration details
                createRestorationDetails(data);
                
                // Show results sections
                comparisonContainer.classList.remove('hidden');
                restorationDetails.classList.remove('hidden');
                resultsContent.classList.remove('hidden');
                resultActions.classList.remove('hidden');
                restoreButton.disabled = false;
            })
            .catch(error => {
                console.error('Error restoring image:', error);
                loadingIndicator.classList.add('hidden');
                comparisonContainer.innerHTML = `
                    <div class="alert alert-error">
                        <span>Error: Could not process your request. Please try again.</span>
                    </div>
                `;
                comparisonContainer.classList.remove('hidden');
                resultsContent.classList.remove('hidden');
                restoreButton.disabled = false;
            });
        });
        
        // Create the before/after comparison slider
        function createComparisonSlider(beforeImgData, afterImgData) {
            const beforeSrc = 'data:image/jpeg;base64,' + beforeImgData;
            const afterSrc = 'data:image/jpeg;base64,' + afterImgData;
            
            const sliderHTML = `
                <h3 class="text-lg font-semibold mb-4 text-center">Original Building → AI Restoration</h3>
                <div class="comparison-slider relative">
                    <div class="before-after-container">
                        <img src="${beforeSrc}" class="before-image" alt="Original Building">
                        <div class="after-container" style="width: 50%;">
                            <img src="${afterSrc}" class="after-image" alt="Restored Building">
                        </div>
                        <div class="slider-handle"></div>
                        <div class="slider-label before-label">Original</div>
                        <div class="slider-label after-label">AI Restored</div>
                    </div>
                </div>
            `;
            
            // Set HTML
            comparisonContainer.innerHTML = sliderHTML;
            
            // Setup slider functionality
            setupSlider();
        }
        
        // Setup the slider functionality
        function setupSlider() {
            const container = document.querySelector('.before-after-container');
            const handle = document.querySelector('.slider-handle');
            const afterContainer = document.querySelector('.after-container');
            
            let isDragging = false;
            
            // Handle mouse events
            handle.addEventListener('mousedown', startDrag);
            document.addEventListener('mousemove', drag);
            document.addEventListener('mouseup', stopDrag);
            
            // Handle touch events
            handle.addEventListener('touchstart', startDrag);
            document.addEventListener('touchmove', drag);
            document.addEventListener('touchend', stopDrag);
            
            function startDrag(e) {
                isDragging = true;
                e.preventDefault();
            }
            
            function drag(e) {
                if (!isDragging) return;
                
                let clientX;
                if (e.type === 'touchmove') {
                    clientX = e.touches[0].clientX;
                } else {
                    clientX = e.clientX;
                }
                
                const rect = container.getBoundingClientRect();
                const x = clientX - rect.left;
                const width = container.offsetWidth;
                
                // Calculate percentage (constrained between 0 and 100)
                let percent = (x / width) * 100;
                percent = Math.max(0, Math.min(100, percent));
                
                // Update elements
                afterContainer.style.width = percent + '%';
                handle.style.left = percent + '%';
            }
            
            function stopDrag() {
                isDragging = false;
            }
        }
        
        // Create restoration details section
        function createRestorationDetails(data) {
            const style = data.style;
            const options = data.options;
            const azureAnalysis = data.azure_analysis;
            
            let featuresHTML = '<ul class="list-disc list-inside text-sm mt-2">';
            
            if (options.preserve_heritage) {
                featuresHTML += '<li>Heritage elements preserved</li>';
            }
            if (options.landscaping) {
                featuresHTML += '<li>Enhanced landscaping and greenery</li>';
            }
            if (options.lighting) {
                featuresHTML += '<li>Architectural lighting highlighted</li>';
            }
            if (options.expand_building) {
                featuresHTML += '<li>Tasteful expansion considered</li>';
            }
            
            featuresHTML += '</ul>';
            
            // Create Azure analysis section
            let analysisHTML = '';
            if (azureAnalysis) {
                analysisHTML = `
                    <div class="mb-4 p-3 bg-accent/10 rounded-lg">
                        <span class="font-semibold text-accent">🤖 AI Building Analysis:</span>
                        <p class="text-sm mt-1">${azureAnalysis}</p>
                    </div>
                `;
            }
            
            // Check if restoration was successful
            let statusInfo = '';
            if (data.restoration_success) {
                statusInfo = `<div class="text-xs text-success mt-4">
                    <p>✨ Powered by Azure OpenAI Image Editing</p>
                </div>`;
            } else {
                statusInfo = `<div class="text-xs text-warning mt-4">
                    <p>⚠️ AI restoration failed - showing original image</p>
                </div>`;
            }
            
            // Create details HTML
            const detailsHTML = `
                <div class="bg-base-200 p-4 rounded-lg">
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="text-lg font-bold">Restoration Details</h3>
                        <span class="badge badge-primary">${style}</span>
                    </div>
                    ${analysisHTML}
                    <div class="mb-2">
                        <span class="font-semibold">Features:</span>
                        ${featuresHTML}
                    </div>
                    ${statusInfo}
                </div>
            `;
            
            // Set HTML
            restorationDetails.innerHTML = detailsHTML;
        }
        
        // Setup download button
        downloadButton.addEventListener('click', function() {
            if (!restoredImageData) return;
            
            // Create download link
            const link = document.createElement('a');
            link.href = 'data:image/jpeg;base64,' + restoredImageData;
            link.download = 'restored_building.jpg';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
        
        // Setup new button
        newButton.addEventListener('click', function() {
            // Reset form
            resetForm();
            
            // Reset results
            resultsPlaceholder.classList.remove('hidden');
            resultsContent.classList.add('hidden');
            resultActions.classList.add('hidden');
            comparisonContainer.classList.add('hidden');
            restorationDetails.classList.add('hidden');
            
            // Reset state
            originalImageData = null;
            restoredImageData = null;
        });
        
        // Set up drag and drop
        const dropzone = document.querySelector('label[for="image-input"]');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, function(e) {
                e.preventDefault();
                e.stopPropagation();
            });
        });
        
        // Highlight on drag
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, function() {
                dropzone.classList.add('bg-base-200');
            });
        });
        
        // Remove highlight on drag leave/drop
        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, function() {
                dropzone.classList.remove('bg-base-200');
            });
        });
        
        // Handle file drop
        dropzone.addEventListener('drop', function(e) {
            const file = e.dataTransfer.files[0];
            
            if (file) {
                // Update file input
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                imageInput.files = dataTransfer.files;
                
                // Trigger change event
                const event = new Event('change');
                imageInput.dispatchEvent(event);
            }
        });
    });
    """)
    
    return Title("Building Restoration Visualizer"), Main(
        form_script,
        Div(
            H1("Building Restoration Visualizer", cls="text-3xl font-bold text-center mb-2 text-arch-blue"),
            P("Powered by Azure OpenAI Image Editing + GPT-4 Analysis", cls="text-center mb-8 text-base-content/70"),
            Div(
                control_panel,
                results_panel,
                cls="flex flex-col md:flex-row gap-6 w-full"
            ),
            cls="container mx-auto px-4 py-8 max-w-6xl"
        ),
        cls="min-h-screen bg-base-100",
        data_theme="light"
    )

# Restoration API Endpoint
@rt("/restore", methods=["POST"])
async def api_restore_building(request):
    """API endpoint to generate building restoration using Azure OpenAI Image Editing + GPT-4 Analysis"""
    try:
        # Get image data and options from request JSON
        data = await request.json()
        image_data = data.get("image_data", "")
        options = data.get("options", {})
        
        if not image_data:
            return JSONResponse({"error": "No image data provided"}, status_code=400)
        
        # Check for API keys
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not azure_api_key or not azure_endpoint:
            return JSONResponse({
                "error": "Azure credentials not found.",
                "help": "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and optionally AZURE_OPENAI_DEPLOYMENT_NAME environment variables"
            }, status_code=401)
        
        # Call the restoration function
        result = restore_building_image(image_data, options)
        
        return JSONResponse(result)
            
    except Exception as e:
        print(f"Error restoring image: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# Debug endpoint to view stored results
@rt("/debug/results")
def debug_results():
    """Debug endpoint to view all stored restoration results"""
    return JSONResponse({
        "total_results": len(restoration_results),
        "cache_size": len(analysis_cache),
        "results": list(restoration_results.keys())
    })

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Building Restoration Visualizer with Azure OpenAI...")
    print(f"📊 In-memory storage initialized")
    print(f"🔑 Azure OpenAI: {'✅' if (os.environ.get('AZURE_OPENAI_API_KEY') and os.environ.get('AZURE_OPENAI_ENDPOINT')) else '❌'}")
    uvicorn.run(app, host="0.0.0.0", port=8002)
