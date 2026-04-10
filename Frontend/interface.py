import streamlit as st
import requests
import io
from PIL import Image
import time
import os


# Configure the page
st.set_page_config(
    page_title="OCR Text Extraction",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .language-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .arabic-badge {
        background-color: #FFE4B5;
        color: #8B4513;
    }
    .french-badge {
        background-color: #E6E6FA;
        color: #4B0082;
    }
    .result-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .error-container {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📄 OCR Text Extraction from Accident Reports </h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Extract text from images in French and Arabic</p>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input(
        "FastAPI Endpoint URL",
        #value="http://localhost:8000/full-pipeline/pipeline_ocr",
        value=os.environ.get("API_URL", "http://localhost:8000/full-pipeline/pipeline_ocr"),
        help="Enter the URL of your FastAPI OCR endpoint"
    )
    st.markdown("---")
    st.markdown("### 📋 Supported Languages")
    st.markdown("🇫🇷 **French** - Latin script")
    st.markdown("🇸🇦 **Arabic** - Arabic script")
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("• Use clear, high-quality images")
    st.markdown("• Ensure good lighting and contrast")
    st.markdown("• Text should be clearly visible")

# Main content layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing French or Arabic text"
    )

    if uploaded_file is not None:
        # Show preview
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown(f"**File name:** {uploaded_file.name}")
        st.markdown(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
        st.markdown(f"**File size:** {uploaded_file.size} bytes")

with col2:
    st.header("🔍 OCR Results")

    if uploaded_file is not None:
        if st.button("🚀 Extract Text", type="primary", use_container_width=True):
            with st.spinner("Processing image... This takes a few moments"):
                try:
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    files = {"file": (uploaded_file.name, io.BytesIO(image_bytes), uploaded_file.type)}

                    start_time = time.time()
                    response = requests.post(api_url, files=files, timeout=3000)
                    processing_time = time.time() - start_time

                    if response.status_code == 200:
                        try:
                            result = response.json()
                        except Exception:
                            st.error("❌ Could not parse JSON response from API.")
                            st.stop()

                        st.success("✅ Text extraction completed!")

                        detected_lang = result.get("detected_language", "unknown")
                        extracted_text = result.get("extracted_text", {})

                        # Language badge
                        if detected_lang == "ar":
                            st.markdown('<div class="language-badge arabic-badge">🇸🇦 Arabic Detected</div>', unsafe_allow_html=True)
                        elif detected_lang == "fr":
                            st.markdown('<div class="language-badge french-badge">🇫🇷 French Detected</div>', unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Language could not be confidently detected.")

                        # Display text or JSON
                        if isinstance(extracted_text, dict):
                            #main_text = extracted_text.get("text", None)
                            main_text = extracted_text.get("corrected_text", None)



                            st.markdown('<div class="result-container">', unsafe_allow_html=True)
                            st.markdown("**📝 Extracted Text (from JSON):**")

                            if main_text:
                                st.text_area(
                                    "Extracted Text",
                                    value=main_text,
                                    height=200,
                                    label_visibility="collapsed",
                                    help="You can copy this text"
                                )

                                # Stats
                                word_count = len(main_text.split())
                                char_count = len(main_text)
                                col_stats1, col_stats2, col_stats3 = st.columns(3)
                                col_stats1.metric("📊 Characters", char_count)
                                col_stats2.metric("📝 Words", word_count)
                                col_stats3.metric("⏱️ Time", f"{processing_time:.1f}s")

                                st.download_button(
                                    label="💾 Download Text",
                                    data=main_text,
                                    file_name=f"extracted_text_{uploaded_file.name.split('.')[0]}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            else:
                                st.info("No 'text' field found inside extracted_text. Showing full JSON instead.")

                            # Move end of container div before the JSON expander
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Show full JSON in expander
                            with st.expander("🔍 Show OCR JSON Details"):
                                st.json(extracted_text)

                        elif isinstance(extracted_text, str) and extracted_text.strip():
                            st.markdown('<div class="result-container">', unsafe_allow_html=True)
                            st.markdown("**📝 Extracted Text:**")
                            st.text_area(
                                "Extracted Text",
                                value=extracted_text,
                                height=200,
                                label_visibility="collapsed",
                                help="You can copy this text"
                            )
                            word_count = len(extracted_text.split())
                            char_count = len(extracted_text)
                            col_stats1, col_stats2, col_stats3 = st.columns(3)
                            col_stats1.metric("📊 Characters", char_count)
                            col_stats2.metric("📝 Words", word_count)
                            col_stats3.metric("⏱️ Time", f"{processing_time:.1f}s")

                            st.download_button(
                                label="💾 Download Text",
                                data=extracted_text,
                                file_name=f"extracted_text_{uploaded_file.name.split('.')[0]}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ No text was extracted from the image.")

                    elif response.status_code == 422:
                        st.markdown('<div class="error-container">', unsafe_allow_html=True)
                        st.error("❌ Invalid image file or format.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        content_type = response.headers.get("content-type", "")
                        if "application/json" in content_type:
                            error_detail = response.json().get("detail", "Unknown error")
                        else:
                            error_detail = response.text
                        st.markdown('<div class="error-container">', unsafe_allow_html=True)
                        st.error(f"❌ API Error ({response.status_code}): {error_detail}")
                        st.markdown('</div>', unsafe_allow_html=True)

                except requests.exceptions.Timeout:
                    st.markdown('<div class="error-container">', unsafe_allow_html=True)
                    st.error("⏱️ Request timed out. Try a smaller image.")
                    st.markdown('</div>', unsafe_allow_html=True)
                except requests.exceptions.ConnectionError:
                    st.markdown('<div class="error-container">', unsafe_allow_html=True)
                    st.error("🔌 Failed to connect to API. Is it running?")
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown('<div class="error-container">', unsafe_allow_html=True)
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Upload an image to begin OCR processing.")

# Footer
st.markdown("---")
