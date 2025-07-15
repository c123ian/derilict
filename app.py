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
            deployment=azure_deployment
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
    """
    Send an image-edit request to Azure OpenAI.
    If the request is blocked by the moderation system (HTTP 400 + code=moderation_blocked)
    we wait two seconds and retry exactly once.  Any other failure—or a second block—
    falls back to the original image data so the app keeps working.
    """
    # ------------------------------------------------------------------
    # 1. Gather Azure config
    # ------------------------------------------------------------------
    endpoint   = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key    = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-image-1")
    api_ver    = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

    if not endpoint or not api_key:
        print("⚠️  Azure OpenAI image credentials missing")
        return original_image_data

    if not endpoint.endswith("/"):
        endpoint += "/"
    url = f"{endpoint}openai/deployments/{deployment}/images/edits?api-version={api_ver}"

    # ------------------------------------------------------------------
    # 2. Build prompt & payload
    # ------------------------------------------------------------------
    prompt = f"""Transform this derelict building into a beautifully restored version
while maintaining EXACTLY the same camera angle, perspective, and viewpoint.

CRITICAL REQUIREMENTS:
- Keep the EXACT same perspective, camera angle, and viewpoint
- Maintain the same architectural proportions and scale
- Preserve the building's structural footprint and shape
- Create photorealistic results that look like professional architectural photography

RESTORATION IMPROVEMENTS:
- Clean and repair all damaged materials
- Replace broken or boarded windows with new glass
- Fresh paint or protective coatings to façades
- Repair and modernize any visible roof elements
- Clean surroundings and add tasteful landscaping
- Subtle modern lighting fixtures highlighting architecture

STYLE SPECIFICATIONS: {description}

OUTPUT REQUIREMENTS:
- Professional architectural photography quality
- Sharp, high-definition details
- Natural lighting matching the original photo
- Realistic materials and textures
- Finished appearance suitable for a design portfolio
"""

    image_bytes = base64.b64decode(original_image_data)

    files = {
        "image": ("building.png", image_bytes, "image/png")
    }
    data = {
        "prompt": prompt,
        "model": deployment,
        "size": "auto",
        "quality": "medium",
        "n": 1
    }
    headers = {
        "api-key": api_key
    }

    # ------------------------------------------------------------------
    # 3. Helper: post once
    # ------------------------------------------------------------------
    def _post_once():
        try:
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=90)
        except Exception as e:
            return False, f"request_error: {e}"

        if resp.status_code == 200:
            payload = resp.json()
            if payload.get("data"):
                return True, payload["data"][0]["b64_json"]
            return False, "no_data_key"
        return False, resp.text  # includes error JSON for 4xx

    # ------------------------------------------------------------------
    # 4. First attempt
    # ------------------------------------------------------------------
    success, result = _post_once()

    # ------------------------------------------------------------------
    # 5. One retry if moderation blocked
    # ------------------------------------------------------------------
    if not success:
        err_code = None
        try:
            err_code = json.loads(result).get("error", {}).get("code")
        except Exception:
            pass

        if err_code == "moderation_blocked":
            print("🔁 Moderation blocked – waiting 2 s and retrying once …")
            time.sleep(2)
            success, result = _post_once()

    # ------------------------------------------------------------------
    # 6. Return / fallback
    # ------------------------------------------------------------------
    if success and result:
        print("✅ High-quality image restoration completed")
        return result

    print(f"❌ Azure image editing failed after retry. Last response: {result}")
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
            restoration_success = restored_img_data != image_data
        except Exception as e:
            print(f"⚠️ Restoration failed: {e}")
            restored_img_data = image_data
            restoration_success = False

        result_data = {
            "id": result_id,
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


# Set up the FastHTML app with updated DaisyUI 5 CDN
app, rt = fast_app(
    hdrs=(
        # Updated to DaisyUI 5 with proper Tailwind CSS
        Script(src="https://cdn.tailwindcss.com"),
        # Then load DaisyUI
        Link(href="https://cdn.jsdelivr.net/npm/daisyui@4.12.10/dist/full.min.css", rel="stylesheet", type="text/css"),
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
            
            /* Enhanced diff component */
            .diff {
                border: 2px solid var(--color-base-300);
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            }
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
    
    # Restoration options panel (removed size dropdown)
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
            # Before/After comparison using DaisyUI diff
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
            
            // State variables
            let originalImageData = null;
            
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
                
                // Validate file type
                if (!file.type.startsWith('image/')) {
                    alert('Please select a valid image file.');
                    resetForm();
                    return;
                }
                
                // Validate file size (max 10MB)
                if (file.size > 10 * 1024 * 1024) {
                    alert('Image size must be less than 10MB.');
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
                
                // Reset results area
                resultsPlaceholder.classList.remove('hidden');
                loadingIndicator.classList.add('hidden');
            }
            
            // Handle restore button click
            restoreButton.addEventListener('click', function() {
                if (!originalImageData) {
                    alert('Please upload an image first.');
                    return;
                }
                
                // Show loading state
                loadingIndicator.classList.remove('hidden');
                resultsPlaceholder.classList.add('hidden');
                restoreButton.disabled = true;
                restoreButton.textContent = 'Generating...';
                
                // Get form options
                const options = getOptions();
                
                // Send request to API
                fetch('/restore', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        image_data: originalImageData,
                        options: options
                    })
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading indicator
                    loadingIndicator.classList.add('hidden');
                    restoreButton.disabled = false;
                    restoreButton.textContent = 'Generate Restoration';
                    
                    if (data.error) {
                        // Show error message
                        showError(data.error, data.help);
                        return;
                    }
                    
                    // Success - redirect to results page
                    if (data.id) {
                        window.location.href = `/results/${data.id}`;
                    } else {
                        showError('No result ID received from server');
                    }
                })
                .catch(error => {
                    console.error('Error restoring image:', error);
                    loadingIndicator.classList.add('hidden');
                    restoreButton.disabled = false;
                    restoreButton.textContent = 'Generate Restoration';
                    showError('Could not process your request. Please try again.');
                });
            });
            
            // Show error message
            function showError(errorMessage, helpText) {
                let fullErrorMessage = `<div class="alert alert-error mb-4">
                    <div>
                        <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>Error: ${errorMessage}</span>
                    </div>
                </div>`;
                
                if (helpText) {
                    fullErrorMessage += `<div class="alert alert-info mb-4">
                        <div>
                            <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                            </svg>
                            <span>${helpText}</span>
                        </div>
                    </div>`;
                    
                    fullErrorMessage += `<div class="alert alert-warning">
                        <div>
                            <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16c-.77.833.192 2.5 1.732 2.5z" />
                            </svg>
                            <div>
                                <strong>Troubleshooting:</strong>
                                <ul class="list-disc list-inside mt-2 text-sm">
                                    <li>Check your Azure OpenAI credentials</li>
                                    <li>Verify your Azure OpenAI deployment supports image editing</li>
                                    <li>Ensure your account has proper billing setup</li>
                                    <li>Try a different image or smaller file size</li>
                                </ul>
                            </div>
                        </div>
                    </div>`;
                }
                
                // Show error in results area
                resultsPlaceholder.innerHTML = fullErrorMessage;
                resultsPlaceholder.classList.remove('hidden');
            }
            
            // Set up drag and drop functionality
            const dropzone = document.querySelector('label[for="image-input"]').parentElement;
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                });
            });
            
            // Visual feedback for drag and drop
            ['dragenter', 'dragover'].forEach(eventName => {
                dropzone.addEventListener(eventName, function() {
                    dropzone.classList.add('border-primary', 'bg-primary/5');
                });
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, function() {
                    dropzone.classList.remove('border-primary', 'bg-primary/5');
                });
            });
            
            // Handle file drop
            dropzone.addEventListener('drop', function(e) {
                const files = e.dataTransfer.files;
                
                if (files.length > 0) {
                    const file = files[0];
                    
                    // Validate file type
                    if (!file.type.startsWith('image/')) {
                        alert('Please drop a valid image file.');
                        return;
                    }
                    
                    // Update file input
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    imageInput.files = dataTransfer.files;
                    
                    // Trigger change event
                    const event = new Event('change');
                    imageInput.dispatchEvent(event);
                }
            });
            
            // Add keyboard navigation
            document.addEventListener('keydown', function(e) {
                // Enter key on restore button
                if (e.key === 'Enter' && document.activeElement === restoreButton) {
                    e.preventDefault();
                    restoreButton.click();
                }
                
                // Escape key to reset form
                if (e.key === 'Escape' && originalImageData) {
                    resetForm();
                }
            });
            
            // Initialize form state
            resetForm();
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

@rt("/results/{result_id}")
def results_page(result_id: str):
    """Display restoration results on a dedicated page"""
    
    if result_id not in restoration_results:
        return Title("Result Not Found"), Main(
            Div(
                H1("Result Not Found", cls="text-2xl font-bold text-center mb-4"),
                P("The requested restoration result could not be found.", cls="text-center mb-4"),
                A("← Back to Home", href="/", cls="btn btn-primary"),
                cls="container mx-auto px-4 py-8 text-center"
            )
        )
    
    result = restoration_results[result_id]
    
    return Title("Restoration Results"), Main(
        Div(
            # Header
            Div(
                H1("Building Restoration Results", cls="text-3xl font-bold text-center mb-2 text-arch-blue"),
                # P(f"Style: {result['style']}", cls="text-center mb-4 text-lg"),
                Div(
                    A("← Back to Home", href="/", cls="btn btn-outline btn-primary mr-4"),
                    Button("Download Result", cls="btn btn-accent", id="download-btn"),
                    cls="text-center mb-8"
                ),
                cls="mb-8"
            ),
            
            # Full-width image comparison
            Div(
                H2("Before & After Comparison", cls="text-xl font-bold text-center mb-6"),
                Div(
                    # Use a proven image comparison library
                    Div(
                        id="comparison-container",
                        cls="w-full max-w-6xl mx-auto"
                    ),
                    cls="mb-8"
                ),
                cls="mb-8"
            ),
            
            # Details section
            Div(
                Div(
                    H3("Restoration Details", cls="text-lg font-bold mb-4"),
                    Div(
                        P(f"Building Analysis: {result['azure_analysis']}", cls="mb-4"),
                        P(f"Generated: {result['created_at']}", cls="text-sm text-base-content/70"),
                        cls="bg-base-200 p-6 rounded-lg"
                    ),
                    cls="max-w-4xl mx-auto"
                ),
                cls="mb-8"
            ),
            
            cls="container mx-auto px-4 py-8"
        ),
        
        # JavaScript for the comparison
        Script(f"""
            document.addEventListener('DOMContentLoaded', () => {{
            /* ------------------------------------------------------------------
                Base-64 images supplied by FastHTML
                ------------------------------------------------------------------ */
            const originalImage = '{result['original_image']}';
            const restoredImage = '{result['restored_image']}';

            /* ------------------------------------------------------------------
                Inject CSS for the clip-path slider
                ------------------------------------------------------------------ */
            const css = `
            .ba-slider {{
            position: relative;
            overflow: hidden;
            height: 70vh;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,.3);
            }}

            .ba-slider img {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            user-select: none;
            pointer-events: none;
            }}

            .img-front {{
            /* Reveal according to --pos (0 – 100 %) */
            clip-path: inset(0 calc(100% - var(--pos,50%)) 0 0);
            }}

            .handle {{
            position: absolute;
            inset: 0 auto 0 var(--pos,50%);
            width: 4px;
            background: #fff;
            transform: translateX(-50%);
            cursor: ew-resize;
            z-index: 10;
            }}

            .handle::before {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 30px;
            height: 30px;
            background: #fff;
            border-radius: 50%;
            box-shadow: 0 2px 10px rgba(0,0,0,.3);
            transform: translate(-50%,-50%);
            }}

            .handle::after {{
            content: '↔';
            position: absolute;
            top: 50%;
            left: 50%;
            font-size: 14px;
            font-weight: 700;
            color: #333;
            transform: translate(-50%,-50%);
            }}
            `;
            const styleTag = document.createElement('style');
            styleTag.textContent = css;
            document.head.appendChild(styleTag);

            /* ------------------------------------------------------------------
                Build the comparison-slider markup
                ------------------------------------------------------------------ */
            const comparisonHTML = `
                <div class="ba-slider" style="--pos:50%;">
                <img src="data:image/jpeg;base64,${{originalImage}}" alt="Original">
                <img src="data:image/jpeg;base64,${{restoredImage}}" class="img-front" alt="Restored">
                <span class="handle" aria-label="Drag to compare"></span>
                </div>
            `;
            const container = document.getElementById('comparison-container');
            container.innerHTML = comparisonHTML;

            /* ------------------------------------------------------------------
                Slider interaction – update CSS variable  --pos
                ------------------------------------------------------------------ */
            const slider = container.querySelector('.ba-slider');
            const handle = slider.querySelector('.handle');

            function setPos(clientX) {{
                const rect = slider.getBoundingClientRect();
                const pct  = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
                slider.style.setProperty('--pos', (pct * 100) + '%');
            }}

            /* Mouse events */
            handle.addEventListener('mousedown', e => {{
                e.preventDefault();
                const move = m => setPos(m.clientX);
                const up   = () => {{
                document.removeEventListener('mousemove', move);
                document.removeEventListener('mouseup',   up);
                }};
                document.addEventListener('mousemove', move);
                document.addEventListener('mouseup',   up);
            }});

            /* Touch events */
            handle.addEventListener('touchstart', () => {{
                const move = t => setPos(t.touches[0].clientX);
                const end  = () => {{
                document.removeEventListener('touchmove', move);
                document.removeEventListener('touchend',  end);
                }};
                document.addEventListener('touchmove', move);
                document.addEventListener('touchend',  end);
            }});

            /* ------------------------------------------------------------------
                Download restored image
                ------------------------------------------------------------------ */
            document.getElementById('download-btn').addEventListener('click', () => {{
                const link = document.createElement('a');
                link.href = 'data:image/jpeg;base64,' + restoredImage;
                link.download = 'restored_building.jpg';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }});
            }});
            """),

        
        cls="min-h-screen bg-base-100"
    )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Building Restoration Visualizer with Azure OpenAI...")
    print(f"📊 In-memory storage initialized")
    print(f"🔑 Azure OpenAI: {'✅' if (os.environ.get('AZURE_OPENAI_API_KEY') and os.environ.get('AZURE_OPENAI_ENDPOINT')) else '❌'}")
    uvicorn.run(app, host="0.0.0.0", port=8002)
