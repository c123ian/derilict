import modal
import os
import sqlite3
import uuid
import time
import json
import base64
import requests
from typing import Optional, Dict, Any, List

from fasthtml.common import *
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse

# Define app
app = modal.App("derelict_restoration")

# Constants and directories
DATA_DIR = "/data"
RESULTS_FOLDER = "/data/restoration_results"
DB_PATH = "/data/derelict_restoration.db"
STATUS_DIR = "/data/status"

# OpenAI API constants
OPENAI_API_KEY = "sk-proj-jxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_VISION_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_IMAGE_API_URL = "https://api.openai.com/v1/images/generations"

# Restoration types
RESTORATION_TYPES = [
    "Home Restoration",
    "Commercial Restoration"
]

# Create custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "requests",
        "python-fasthtml==0.12.0"
    )
)

# Look up data volume for storing results
try:
    derelict_volume = modal.Volume.lookup("derelict_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    derelict_volume = modal.Volume.persisted("derelict_volume")

# Base prompt template for OpenAI
RESTORATION_PROMPT = """
Create a photorealistic restoration of this derelict building, showing how it would look beautifully renovated and restored. 
{restoration_type_instructions}

Maintain the same architectural style, building position, perspective, and surroundings, but transform the building into a pristine, restored condition.

Show:
- Repaired walls with fresh paint or restored original materials
- New windows and doors
- Fixed roof
- Clean and well-maintained exterior
- Attractive landscaping
- Appropriate lighting
- Overall appealing aesthetic that respects the original structure
"""

HOME_INSTRUCTIONS = """
Style it as a beautiful residential home with:
- Warm, inviting appearance
- Residential-appropriate colors and finishes
- Cozy exterior lighting
- Home-style landscaping with garden elements
- Suitable residential details like a mailbox, porch furniture, etc.
"""

COMMERCIAL_INSTRUCTIONS = """
Style it as an attractive commercial building with:
- Professional, polished appearance
- Business-appropriate signage (generic/neutral)
- Commercial-grade windows and doors
- Professional landscaping
- Exterior lighting suitable for a business
- Clean, accessible entrance area
"""

# Function to save results to file
def save_results_file(result_id, original_image, restored_image, result_content):
    """Save restoration results to a file"""
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    result_file = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
    result_data = {
        "id": result_id,
        "result": result_content,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open(result_file, "w") as f:
            json.dump(result_data, f)
        
        # Save the original and restored images
        with open(os.path.join(RESULTS_FOLDER, f"{result_id}_original.jpg"), "wb") as f:
            f.write(base64.b64decode(original_image))
        
        with open(os.path.join(RESULTS_FOLDER, f"{result_id}_restored.jpg"), "wb") as f:
            f.write(base64.b64decode(restored_image))
            
        print(f"✅ Saved result files for ID: {result_id}")
        return True
    except Exception as e:
        print(f"⚠️ Error saving result files: {e}")
        return False

# Setup database for restoration results
def setup_database(db_path: str):
    """Initialize SQLite database for restoration results"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    
    # Enable WAL mode for better concurrency
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            restoration_type TEXT NOT NULL,
            prompt TEXT NOT NULL,
            original_image_path TEXT NOT NULL,
            restored_image_path TEXT NOT NULL,
            status TEXT DEFAULT 'generated',
            feedback TEXT DEFAULT NULL, 
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

# Generate restoration using OpenAI's API
@app.function(
    image=image,
    cpu=1.0,
    timeout=300,
    volumes={DATA_DIR: derelict_volume}
)
def generate_restoration(image_data: str, options: Dict[str, bool]) -> Dict[str, Any]:
    """
    Generate restored building image using OpenAI's API based on provided options
    
    Args:
        image_data: Base64 encoded image
        options: Dictionary of toggle options
    
    Returns:
        Dictionary with restoration results
    """
    result_id = uuid.uuid4().hex
    
    # Build prompt based on options
    restoration_type = "Home Restoration" if options.get("home_restoration", True) else "Commercial Restoration"
    
    # Select the appropriate instructions
    if restoration_type == "Home Restoration":
        restoration_type_instructions = HOME_INSTRUCTIONS
    else:
        restoration_type_instructions = COMMERCIAL_INSTRUCTIONS
    
    # Prepare the full prompt
    prompt = RESTORATION_PROMPT.format(
        restoration_type_instructions=restoration_type_instructions
    )
    
    print(f"🔍 Sending image to OpenAI for {restoration_type.lower()}...")
    
    try:
        # Format the image data properly
        if "," in image_data:
            # If it contains a comma, it's likely in the format "data:image/jpeg;base64,/9j/..."
            # We need to extract just the base64 part
            image_data_full = image_data
            image_data = image_data.split(",", 1)[1]
            
        print("🔍 Step 1: Analyzing building image with GPT-4V...")
        
        # First, use GPT-4V to analyze the building and generate a detailed description
        vision_headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        vision_payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert architect specializing in building restoration. Analyze this derelict building image and provide a detailed description of its architectural style, key features, materials, and surroundings. Your description will be used to generate a restoration image."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this derelict building in detail. Focus on architectural elements, layout, materials, surroundings, and style. Be specific about features that would need to be restored."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
        
        # Make the vision API call
        vision_response = requests.post(OPENAI_VISION_API_URL, headers=vision_headers, json=vision_payload)
        vision_response.raise_for_status()
        
        # Extract the building description
        vision_result = vision_response.json()
        building_description = vision_result["choices"][0]["message"]["content"]
        
        print("✅ Building analysis complete")
        print("🎨 Step 2: Generating restored building image...")
        
        # Now, use the building description with the selected restoration style
        # to create a prompt for DALL-E
        
        # Create a detailed prompt combining the building description and restoration type
        enhanced_prompt = f"""Create a photorealistic image of this restored building:

{building_description}

{prompt}

Important: Maintain the exact same architectural style, building position, perspective, and surroundings as the original building. Only show the transformation from derelict to pristine condition.
"""
        
        # Prepare the DALL-E API request
        generation_headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        generation_payload = {
            "model": "dall-e-3",
            "prompt": enhanced_prompt,
            "n": 1,
            "size": "1024x1024",
            "response_format": "b64_json",
            "quality": "hd"
        }
        
        # Make the image generation API call
        generation_response = requests.post(OPENAI_IMAGE_API_URL, headers=generation_headers, json=generation_payload)
        generation_response.raise_for_status()
        
        # Extract the response content
        result = generation_response.json()
        restored_image_b64 = result["data"][0]["b64_json"]
        
        # Store the results in the database
        try:
            conn = setup_database(DB_PATH)
            cursor = conn.cursor()
            
            # Save paths to images
            original_path = os.path.join(RESULTS_FOLDER, f"{result_id}_original.jpg")
            restored_path = os.path.join(RESULTS_FOLDER, f"{result_id}_restored.jpg")
            
            cursor.execute(
                "INSERT INTO results (id, restoration_type, prompt, original_image_path, restored_image_path) VALUES (?, ?, ?, ?, ?)",
                (result_id, restoration_type, prompt, original_path, restored_path)
            )
            
            conn.commit()
            conn.close()
            
            # Save results to file
            save_results_file(result_id, image_data, restored_image_b64, {
                "restoration_type": restoration_type,
                "prompt": prompt,
                "building_description": building_description
            })
            
        except Exception as e:
            print(f"⚠️ Error saving to database: {e}")
            raise e
        
        return {
            "id": result_id,
            "restoration_type": restoration_type,
            "original_image": image_data,
            "restored_image": restored_image_b64,
            "prompt": prompt,
            "building_description": building_description
        }
        
    except Exception as e:
        print(f"⚠️ Error generating restoration: {e}")
        return {
            "error": str(e),
            "id": result_id
        }

# Main FastHTML Server with defined routes
@app.function(
    image=image,
    volumes={DATA_DIR: derelict_volume},
    cpu=1.0,
    timeout=3600
)
@modal.asgi_app()
def serve():
    """Main FastHTML Server for Derelict Building Restoration Visualizer"""
    # Set up the FastHTML app with required headers
    fasthtml_app, rt = fast_app(
        hdrs=(
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@3.9.2/dist/full.css"),
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
            Script(src="https://unpkg.com/htmx.org@1.9.10"),
            # Add custom theme styles from https://daisyui.com/theme-generator/
            Style("""
                :root {
                --color-base-100: oklch(98% 0.002 247.839);
                --color-base-200: oklch(96% 0.003 264.542);
                --color-base-300: oklch(92% 0.006 264.531);
                --color-base-content: oklch(21% 0.034 264.665);
                --color-primary: oklch(47% 0.266 120.957);  /* Green for sustainability */
                --color-primary-content: oklch(97% 0.014 254.604);
                --color-secondary: oklch(74% 0.234 93.635);  /* Yellow for construction */
                --color-secondary-content: oklch(13% 0.028 261.692);
                --color-accent: oklch(41% 0.234 41.252);     /* Brown accent */
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
                --radius-selector: 0rem;
                --radius-field: 0.5rem;
                --radius-box: 2rem;
                --size-selector: 0.25rem;
                --size-field: 0.25rem;
                --border: 1px;
                }

                /* Custom styling for better contrast */
                .text-restoration-green {
                    color: oklch(47% 0.266 120.957);
                }
                
                .bg-restoration-yellow {
                    background-color: oklch(74% 0.234 93.635);
                }
                
                .custom-border {
                    border-color: var(--color-base-300);
                }
                
                /* Custom styles for the diff slider */
                .diff {
                  position: relative;
                  display: inline-block;
                  overflow: hidden;
                  margin: 0;
                  width: 100%;
                }
                
                .diff-item-1,
                .diff-item-2 {
                  position: relative;
                  width: 100%;
                  height: 100%;
                }
                
                .diff-item-1 img,
                .diff-item-2 img {
                  width: 100%;
                  height: 100%;
                  object-fit: cover;
                  object-position: left;
                }
                
                .diff-item-2 {
                  position: absolute;
                  overflow: hidden;
                  top: 0;
                  width: 50%;
                }
                
                .diff-resizer {
                  position: absolute;
                  width: 4px;
                  height: calc(100% - 16px);
                  top: 8px;
                  right: calc(50% - 2px);
                  background-color: white;
                  box-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
                  cursor: col-resize;
                  z-index: 30;
                }
                
                .diff::before {
                  content: "Before";
                  position: absolute;
                  left: 8px;
                  top: 8px;
                  background-color: rgba(255, 255, 255, 0.85);
                  padding: 4px 8px;
                  border-radius: 4px;
                  font-size: 12px;
                  z-index: 20;
                }
                
                .diff::after {
                  content: "After";
                  position: absolute;
                  right: 8px;
                  top: 8px;
                  background-color: rgba(255, 255, 255, 0.85);
                  padding: 4px 8px;
                  border-radius: 4px;
                  font-size: 12px;
                  z-index: 20;
                }
                
                /* Loading animation */
                .loading-progress {
                  width: 120px;
                  height: 24px;
                  -webkit-mask: linear-gradient(90deg, #000 70%, #0000 0) left/20% 100%;
                  background: linear-gradient(#000 0 0) left/0% 100% no-repeat #ddd;
                  animation: loading-progress-animation 2s infinite steps(6);
                }
                
                @keyframes loading-progress-animation {
                  100% {background-size: 120% 100%}
                }
            """),
        )
    )
    
    # Ensure database exists
    setup_database(DB_PATH)
    
    #################################################
    # Homepage Route - Derelict Building Restoration Dashboard
    #################################################
    @rt("/")
    def homepage():
        """Render the derelict building restoration dashboard"""
        
        # Image upload section with HTMX to preview the image
        image_upload = Div(
            Label("Upload Derelict Building Image", cls="block text-xl font-medium mb-2 text-restoration-green"),
            P("Upload an image of a derelict building to visualize how it would look if restored.", cls="mb-4"),
            Div(
                Label(
                    Div(
                        Span("Click or drag image here", cls="text-lg text-center"),
                        P("Supported formats: JPEG, PNG", cls="text-sm text-center mt-2"),
                        cls="flex flex-col items-center justify-center h-full"
                    ),
                    Input(
                        type="file",
                        name="building_image",
                        accept="image/jpeg,image/png",
                        cls="hidden",
                        id="image-input",
                        hx_on="change: showImagePreview(event)"
                    ),
                    cls="w-full h-40 border-2 border-dashed rounded-lg flex items-center justify-center cursor-pointer hover:bg-base-200 transition-colors",
                    id="dropzone"
                ),
                cls="mb-6"
            ),
            Div(
                Img(
                    id="image-preview",
                    src="",
                    cls="max-h-64 mx-auto hidden object-contain rounded-lg border shadow-sm"
                ),
                cls="mb-6",
                id="preview-container"
            ),
            cls="mb-8"
        )
        
        # Restoration options with HTMX
        restoration_options = Div(
            H3("Restoration Options", cls="text-lg font-semibold mb-4 text-restoration-green"),
            Div(
                Label(
                    Input(
                        type="radio",
                        name="restoration_type",
                        value="home",
                        checked="checked",
                        cls="radio radio-primary mr-3"
                    ),
                    Span("Home Restoration"),
                    cls="label cursor-pointer justify-start"
                ),
                cls="mb-3"
            ),
            Div(
                Label(
                    Input(
                        type="radio",
                        name="restoration_type",
                        value="commercial",
                        cls="radio radio-primary mr-3"
                    ),
                    Span("Commercial Restoration"),
                    cls="label cursor-pointer justify-start"
                ),
                cls="mb-3"
            ),
            cls="mb-6 p-4 bg-base-200 rounded-lg"
        )
        
        # Form with HTMX for submission
        restoration_form = Form(
                image_upload,
                restoration_options,
                Button(
                    "Generate Restoration",
                    cls="btn btn-primary w-full",
                    id="restore-button",
                    disabled="disabled",
                    type="submit"
                ),
                id="restoration-form",
                hx_post="/restore",
                hx_trigger="submit",
                hx_target="#results-container",
                hx_swap="innerHTML",
                hx_indicator="#loading-container",
                hx_on="htmx:beforeRequest: document.getElementById('loading-container').classList.remove('hidden');"
            )
        
        # Controls panel
        controls_panel = Div(
            H2("Derelict Building Restoration", cls="text-xl font-bold mb-4 text-restoration-green"),
            restoration_form,
            cls="w-full md:w-1/2 bg-base-100 p-6 rounded-lg shadow-lg custom-border border"
        )
        
        # Results panel
        results_panel = Div(
            H2("Restoration Results", cls="text-xl font-bold mb-4 text-restoration-green"),
            Div(
                Div(
                    cls="loading-progress mx-auto",
                ),
                P("Generating your restoration...", cls="text-center mt-4 text-base-content/70"),
                cls="flex flex-col justify-center items-center h-32 hidden",
                id="loading-container",
                hx_swap_oob="true"
            ),
            Div(
                P("Upload an image and click 'Generate Restoration' to see results.", 
                  cls="text-center text-base-content/70 italic"),
                id="results-placeholder",
                cls="text-center py-12"
            ),
            Div(
                id="results-container",
                cls="w-full"
            ),
            cls="w-full md:w-1/2 bg-base-100 p-6 rounded-lg shadow-lg custom-border border"
        )
        
        # Add minimal JavaScript for image preview and diff slider
        js_script = Script("""
            // Image preview function
            function showImagePreview(event) {
                const file = event.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const preview = document.getElementById('image-preview');
                        preview.src = e.target.result;
                        preview.classList.remove('hidden');
                        
                        // Enable the restore button
                        document.getElementById('restore-button').disabled = false;
                        
                        // Store the image data in a hidden input for submission
                        let imageDataInput = document.getElementById('image-data-input');
                        if (!imageDataInput) {
                            imageDataInput = document.createElement('input');
                            imageDataInput.type = 'hidden';
                            imageDataInput.name = 'image_data';
                            imageDataInput.id = 'image-data-input';
                            document.getElementById('restoration-form').appendChild(imageDataInput);
                        }
                        imageDataInput.value = e.target.result;
                    };
                    reader.readAsDataURL(file);
                }
            }
            
            // Initialize diff slider 
            function initDiffSlider() {
                const sliders = document.querySelectorAll('.diff');
                
                sliders.forEach(slider => {
                    const resizer = slider.querySelector('.diff-resizer');
                    const item2 = slider.querySelector('.diff-item-2');
                    
                    if (!resizer || !item2) return;
                    
                    let isResizing = false;
                    
                    // Mouse events
                    resizer.addEventListener('mousedown', function(e) {
                        isResizing = true;
                        e.preventDefault();
                    });
                    
                    document.addEventListener('mousemove', function(e) {
                        if (!isResizing) return;
                        
                        const rect = slider.getBoundingClientRect();
                        const x = e.clientX - rect.left;
                        const percent = (x / rect.width) * 100;
                        
                        // Limit between 5% and 95%
                        const limitedPercent = Math.min(Math.max(percent, 5), 95);
                        
                        item2.style.width = limitedPercent + '%';
                        resizer.style.right = (100 - limitedPercent) + '%';
                    });
                    
                    document.addEventListener('mouseup', function() {
                        isResizing = false;
                    });
                    
                    // Touch events for mobile
                    resizer.addEventListener('touchstart', function(e) {
                        isResizing = true;
                    });
                    
                    document.addEventListener('touchmove', function(e) {
                        if (!isResizing) return;
                        
                        const touch = e.touches[0];
                        const rect = slider.getBoundingClientRect();
                        const x = touch.clientX - rect.left;
                        const percent = (x / rect.width) * 100;
                        
                        // Limit between 5% and 95%
                        const limitedPercent = Math.min(Math.max(percent, 5), 95);
                        
                        item2.style.width = limitedPercent + '%';
                        resizer.style.right = (100 - limitedPercent) + '%';
                    });
                    
                    document.addEventListener('touchend', function() {
                        isResizing = false;
                    });
                });
            }
            
            // Add form validation before submit
            function validateForm(event) {
                // Get the image data input value
                const imageDataInput = document.getElementById('image-data-input');
                
                // Check if image data exists
                if (!imageDataInput || !imageDataInput.value) {
                    // Prevent form submission
                    event.preventDefault();
                    event.stopPropagation();
                    
                    // Show an error message
                    const resultsContainer = document.getElementById('results-container');
                    resultsContainer.innerHTML = `
                        <div class="alert alert-error">
                            <span>Please upload an image before generating a restoration.</span>
                        </div>
                    `;
                    
                    return false;
                }
                
                // Show loading indicator and continue with submission
                document.getElementById('loading-container').classList.remove('hidden');
                document.getElementById('results-placeholder') && document.getElementById('results-placeholder').classList.add('hidden');
                return true;
            }
            
            // Setup drag and drop for image upload
            function setupDragAndDrop() {
                const dropzone = document.getElementById('dropzone');
                
                if (dropzone) {
                    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                        dropzone.addEventListener(eventName, preventDefaults, false);
                    });
                    
                    function preventDefaults(e) {
                        e.preventDefault();
                        e.stopPropagation();
                    }
                    
                    ['dragenter', 'dragover'].forEach(eventName => {
                        dropzone.addEventListener(eventName, highlight, false);
                    });
                    
                    ['dragleave', 'drop'].forEach(eventName => {
                        dropzone.addEventListener(eventName, unhighlight, false);
                    });
                    
                    function highlight() {
                        dropzone.classList.add('bg-base-200');
                    }
                    
                    function unhighlight() {
                        dropzone.classList.remove('bg-base-200');
                    }
                    
                    dropzone.addEventListener('drop', handleDrop, false);
                    
                    function handleDrop(e) {
                        const dt = e.dataTransfer;
                        const files = dt.files;
                        
                        if (files.length > 0) {
                            const fileInput = document.getElementById('image-input');
                            fileInput.files = files;
                            
                            // Trigger the change event to show preview
                            const event = new Event('change');
                            fileInput.dispatchEvent(event);
                        }
                    }
                }
            }
            
            // Initialize app when DOM is loaded
            document.addEventListener('DOMContentLoaded', function() {
                // Disable the restore button initially
                const restoreButton = document.getElementById('restore-button');
                if (restoreButton) {
                    restoreButton.disabled = true;
                }
                
                // Ensure the loading container is hidden
                const loadingContainer = document.getElementById('loading-container');
                if (loadingContainer) {
                    loadingContainer.classList.add('hidden');
                }
                
                // Setup the form validation
                const restorationForm = document.getElementById('restoration-form');
                if (restorationForm) {
                    restorationForm.addEventListener('submit', validateForm);
                }
                
                // Setup image input handling
                const imageInput = document.getElementById('image-input');
                if (imageInput) {
                    imageInput.addEventListener('change', showImagePreview);
                }
                
                // Setup drag and drop
                setupDragAndDrop();
                
                // Initialize any existing diff sliders
                initDiffSlider();
                
                // Setup a MutationObserver to initialize diff sliders that get added to the DOM
                const observer = new MutationObserver(function(mutations) {
                    mutations.forEach(function(mutation) {
                        if (mutation.type === 'childList' && mutation.addedNodes.length) {
                            mutation.addedNodes.forEach(function(node) {
                                if (node.nodeType === 1 && node.querySelector) {
                                    const newSliders = node.querySelectorAll('.diff');
                                    if (newSliders.length) {
                                        setTimeout(initDiffSlider, 100); // Small delay to ensure DOM is ready
                                    }
                                }
                            });
                        }
                    });
                });
                
                observer.observe(document.body, { childList: true, subtree: true });
            });
        """)
        
        return Title("Derelict Building Restoration"), Main(
            js_script,
            Div(
                H1("Derelict Building Restoration Visualizer", cls="text-3xl font-bold text-center mb-2 text-restoration-green"),
                P("Powered by OpenAI's Image Generation", cls="text-center mb-8 text-base-content/70"),
                Div(
                    controls_panel,
                    results_panel,
                    cls="flex flex-col md:flex-row gap-6 w-full"
                ),
                cls="container mx-auto px-4 py-8 max-w-6xl"
            ),
            cls="min-h-screen bg-base-100",
            data_theme="light"
        )
    
    #################################################
    # Restoration API Endpoint (HTMX Compatible)
    #################################################
    @rt("/restore", methods=["POST"])
    async def api_restore_image(request):
        """API endpoint to generate restored building using OpenAI"""
        try:
            # Get form data
            form_data = await request.form()
            image_data = form_data.get("image_data", "")
            restoration_type = form_data.get("restoration_type", "home")
            
            # Check if we have image data
            if not image_data:
                return HTMLResponse("""
                    <div class="alert alert-error">
                        <span>Error: No image data provided</span>
                    </div>
                """)
            
            # Set up options
            options = {
                "home_restoration": restoration_type == "home",
                "commercial_restoration": restoration_type == "commercial"
            }
            
            # Generate restoration
            result = generate_restoration.remote(image_data, options)
            
            # If there's an error
            if "error" in result:
                return HTMLResponse(f"""
                    <div class="alert alert-error">
                        <span>Error: {result["error"]}</span>
                    </div>
                """)
            
            # Create the result HTML with the diff slider
            restoration_html = f"""
                <div class="mb-6">
                    <div class="diff aspect-16/9 rounded-lg shadow-lg" tabindex="0">
                        <div class="diff-item-1" role="img" tabindex="0">
                            <img alt="Original building" src="data:image/jpeg;base64,{result['original_image']}" />
                        </div>
                        <div class="diff-item-2" role="img">
                            <img alt="Restored building" src="data:image/jpeg;base64,{result['restored_image']}" />
                        </div>
                        <div class="diff-resizer"></div>
                    </div>
                </div>
                
                <div class="p-4 bg-base-200 rounded-lg mb-4">
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="text-lg font-bold">Restoration Type</h3>
                        <span class="badge badge-primary">{result['restoration_type']}</span>
                    </div>
                    <div class="mt-4">
                        <span class="font-semibold">Building Description:</span>
                        <p class="mt-2 text-sm">{result['building_description']}</p>
                    </div>
                </div>
                
                <div class="mt-6 flex justify-end items-center gap-2">
                    <a class="btn btn-outline btn-accent btn-sm" 
                       href="data:image/jpeg;base64,{result['restored_image']}" 
                       download="restored-building.jpg">
                        Download Restored Image
                    </a>
                    <button class="btn btn-outline btn-primary btn-sm"
                            hx-get="/"
                            hx-push-url="true">
                        New Restoration
                    </button>
                </div>
            """
            
            return HTMLResponse(restoration_html)
                
        except Exception as e:
            print(f"Error generating restoration: {e}")
            return HTMLResponse(f"""
                <div class="alert alert-error">
                    <span>Error: {str(e)}</span>
                </div>
            """)
    
    # Return the FastHTML app
    return fasthtml_app

if __name__ == "__main__":
    print("Starting Derilict App...")
