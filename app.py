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
app = modal.App("building_restoration")

# Define secret to access OpenAI API key
openai_secret = modal.Secret.from_name("openai-api-key")

# Constants and directories
DATA_DIR = "/data"
RESULTS_FOLDER = "/data/restoration_results"
DB_PATH = "/data/building_restoration.db"
STATUS_DIR = "/data/status"

# OpenAI API URLs
OPENAI_GENERATIONS_URL = "https://api.openai.com/v1/images/generations"
OPENAI_EDITS_URL = "https://api.openai.com/v1/images/edits"

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
    building_volume = modal.Volume.lookup("building_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    building_volume = modal.Volume.persisted("building_volume")

# Base prompt template for OpenAI GPT Image
RESTORATION_PROMPT = """
Create a realistic visualization of a derelict building after professional restoration and renovation.
{style_instruction}
{additional_instructions}
Maintain the same architectural footprint and core structure, but repair all damage.
Fix broken windows, repair the facade, update the exterior, and modernize the appearance while respecting the building's original character.
Make the surrounding area clean and well-maintained.
The result should look like a professional architectural visualization of the restored building.
"""

# Function to save results to file
def save_results_file(result_id, result_content):
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
        print(f"✅ Saved result file for ID: {result_id}")
        return True
    except Exception as e:
        print(f"⚠️ Error saving result file: {e}")
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
    
    # Create tables for restoration results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            style TEXT NOT NULL,
            prompt TEXT NOT NULL,
            original_image TEXT NOT NULL,
            restored_image TEXT NOT NULL,
            additional_details TEXT,
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
    volumes={DATA_DIR: building_volume},
    secrets=[openai_secret]  # Use the Modal secret
)
def restore_building_image(image_data: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate building restoration using OpenAI's GPT Image API
    
    Args:
        image_data: Base64 encoded image
        options: Dictionary of restoration options
    
    Returns:
        Dictionary with restoration results
    """
    # Now the secret is available as an environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Log if we have an API key (don't print the actual key)
    print(f"🔑 API key available: {api_key is not None and api_key != ''}")
    
    if not api_key:
        return {
            "error": "OpenAI API key not found in environment variables or Modal secrets.",
            "help": "Please create a Modal secret with 'modal secret create openai-api-key OPENAI_API_KEY=your-key'."
        }
    
    result_id = uuid.uuid4().hex
    
    # Get selected style
    selected_style = options.get("style", "Modern renovation")
    style_instruction = f"Use a {selected_style} style for the restoration."
    
    # Build additional instructions based on options
    additional_instructions = []
    
    if options.get("preserve_heritage", False):
        additional_instructions.append("Preserve historical and heritage elements of the building.")
        
    if options.get("landscaping", False):
        additional_instructions.append("Add attractive landscaping and greenery around the building.")
        
    if options.get("lighting", False):
        additional_instructions.append("Add modern and attractive lighting to highlight architectural features.")
    
    if options.get("expand_building", False):
        additional_instructions.append("Consider a tasteful expansion or addition that complements the original structure.")
    
    # Prepare the prompt
    additional_instructions_text = " ".join(additional_instructions) if additional_instructions else ""
    
    prompt = RESTORATION_PROMPT.format(
        style_instruction=style_instruction,
        additional_instructions=additional_instructions_text
    )
    
    print("🔍 Sending image to OpenAI for restoration visualization...")
    
    try:
        # Save original image for comparison
        original_img_data = image_data
        
        # Decode base64 to binary
        image_binary = base64.b64decode(image_data)
        
        # Prepare the request for OpenAI API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print("🔍 Using OpenAI API: generations endpoint")
        
        # First try the generations endpoint as a fallback since edits might require special permissions
        payload = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "high",
            "response_format": "b64_json"
        }
        
        try:
            # Try to use the generations endpoint first
            print("🔄 Attempting to use the generations endpoint...")
            response = requests.post(
                OPENAI_GENERATIONS_URL, 
                headers=headers, 
                json=payload
            )
            response.raise_for_status()
            
            # Success with generations
            result = response.json()
            print("✅ Generations endpoint successful")
            
        except requests.exceptions.RequestException as e:
            # If generations fails, try to use the edits endpoint
            print(f"⚠️ Generations endpoint failed: {e}")
            print("🔄 Falling back to edits endpoint...")
            
            try:
                # Create multipart form data for edits
                files = {
                    'image': ('image.jpg', image_binary, 'image/jpeg'),
                    'prompt': (None, prompt),
                    'model': (None, 'gpt-image-1'),
                    'n': (None, '1'),
                    'size': (None, 'auto'),
                    'quality': (None, 'high')
                }
                
                response = requests.post(
                    OPENAI_EDITS_URL, 
                    headers={"Authorization": f"Bearer {api_key}"}, 
                    files=files
                )
                response.raise_for_status()
                result = response.json()
                print("✅ Edits endpoint successful")
                
            except requests.exceptions.RequestException as e:
                error_details = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_json = e.response.json()
                        if 'error' in error_json:
                            error_details = f"{error_json['error'].get('message', str(e))}"
                    except:
                        pass
                
                raise Exception(f"API Error: {error_details}. Please verify your API key has the proper permissions.")
        
        # Extract the response content from result
        print(f"📊 API response structure: {list(result.keys())}")
        
        # Get the restored image
        if 'data' in result and len(result['data']) > 0:
            if 'b64_json' in result['data'][0]:
                print("✅ Received base64 image data")
                restored_img_data = result['data'][0]['b64_json']
            else:
                # If image URL is returned instead of base64
                print("✅ Received image URL, fetching content...")
                img_url = result['data'][0]['url']
                img_response = requests.get(img_url)
                img_response.raise_for_status()
                restored_img_data = base64.b64encode(img_response.content).decode('utf-8')
        else:
            print(f"⚠️ Unexpected API response format: {result}")
            raise Exception("No image data returned from API")
        
        # Store the result in the database
        try:
            conn = setup_database(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO results (id, style, prompt, original_image, restored_image, additional_details) VALUES (?, ?, ?, ?, ?, ?)",
                (result_id, selected_style, prompt, original_img_data, restored_img_data, json.dumps(options))
            )
            
            conn.commit()
            conn.close()
            
            # Save results to file
            save_results_file(result_id, {
                "style": selected_style,
                "prompt": prompt,
                "options": options
            })
            
        except Exception as e:
            print(f"⚠️ Error saving to database: {e}")
            raise e
        
        return {
            "id": result_id,
            "style": selected_style,
            "prompt": prompt,
            "original_image": original_img_data,
            "restored_image": restored_img_data,
            "options": options,
            "usage": result.get("usage", {})
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
    volumes={DATA_DIR: building_volume},
    cpu=1.0,
    timeout=3600,
    secrets=[openai_secret]  # Add the secret here too
)
@modal.asgi_app()
def serve():
    """Main FastHTML Server for Building Restoration Dashboard"""
    # Set up the FastHTML app with required headers
    fasthtml_app, rt = fast_app(
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
                --color-primary: oklch(47% 0.196 209.957);  /* Blue for architecture */
                --color-primary-content: oklch(97% 0.014 254.604);
                --color-secondary: oklch(74% 0.134 119.635);  /* Green for renewal */
                --color-secondary-content: oklch(13% 0.028 261.692);
                --color-accent: oklch(71% 0.134 41.252);     /* Tan accent for buildings */
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

                /* Custom styling */
                .text-arch-blue {
                    color: oklch(47% 0.196 209.957);
                }
                
                .bg-renew-green {
                    background-color: oklch(74% 0.134 119.635);
                }
                
                .custom-border {
                    border-color: var(--color-base-300);
                }

                /* Comparison slider */
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
                
                .before-image,
                .after-image {
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
                
                .before-label {
                    left: 10px;
                }
                
                .after-label {
                    right: 10px;
                }
            """),
        )
    )
    
    # Ensure database exists
    setup_database(DB_PATH)
    
    #################################################
    # Homepage Route - Building Restoration Dashboard
    #################################################
    @rt("/")
    def homepage():
        """Render the building restoration dashboard"""
        
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
            
            // Check for demo mode (if no API key is available)
            const isDemoMode = false; // This can be set server-side if needed
            
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
                                    <li>Make sure your OpenAI API key is set in the environment variables</li>
                                    <li>Verify your API key has access to the GPT Image API (may require subscription)</li>
                                    <li>Check that your OpenAI account has billing information set up</li>
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
                    <h3 class="text-lg font-semibold mb-4 text-center">Before & After Comparison</h3>
                    <div class="comparison-slider">
                        <div class="before-after-container">
                            <img src="${beforeSrc}" class="before-image" alt="Original Building">
                            <div class="after-container" style="width: 50%;">
                                <img src="${afterSrc}" class="after-image" alt="Restored Building">
                            </div>
                            <div class="slider-handle"></div>
                            <div class="slider-label before-label">Before</div>
                            <div class="slider-label after-label">After</div>
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
                
                // Create usage details if available
                let usageHTML = '';
                if (data.usage) {
                    usageHTML = `
                        <div class="text-xs text-base-content/70 mt-4">
                            <p>Tokens used: ${data.usage.total_tokens || 'N/A'}</p>
                        </div>
                    `;
                }
                
                // Create details HTML
                const detailsHTML = `
                    <div class="bg-base-200 p-4 rounded-lg">
                        <div class="flex justify-between items-center mb-2">
                            <h3 class="text-lg font-bold">Restoration Details</h3>
                            <span class="badge badge-primary">${style}</span>
                        </div>
                        <div class="mb-2">
                            <span class="font-semibold">Features:</span>
                            ${featuresHTML}
                        </div>
                        <div class="mb-2">
                            <span class="font-semibold">Prompt Used:</span>
                            <p class="text-sm mt-1">${data.prompt}</p>
                        </div>
                        ${usageHTML}
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
                P("Powered by OpenAI's GPT Image AI", cls="text-center mb-8 text-base-content/70"),
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
    
    #################################################
    # Restoration API Endpoint
    #################################################
    @rt("/restore", methods=["POST"])
    async def api_restore_building(request):
        """API endpoint to generate building restoration using OpenAI"""
        try:
            # Get image data and options from request JSON
            data = await request.json()
            image_data = data.get("image_data", "")
            options = data.get("options", {})
            
            if not image_data:
                return JSONResponse({"error": "No image data provided"}, status_code=400)
            
            # Check for API key - the secret should be loaded into env vars
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                # Add detailed diagnostics
                env_vars = {k: "PRESENT" if v else "MISSING" for k, v in os.environ.items() 
                           if k.startswith("OPENAI") or k.endswith("KEY") or k == "PATH"}
                
                return JSONResponse({
                    "error": "OpenAI API key not found in environment variables.",
                    "help": "You need to create a Modal secret with 'modal secret create openai-api-key OPENAI_API_KEY=your-key'",
                    "debug_info": {
                        "env_diagnostic": env_vars
                    }
                }, status_code=401)
            
            # Call the restoration function
            result = restore_building_image.remote(image_data, options)
            
            return JSONResponse(result)
                
        except Exception as e:
            print(f"Error restoring image: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    # Return the FastHTML app
    return fasthtml_app

if __name__ == "__main__":
    print("Starting Building Restoration Visualizer...")
