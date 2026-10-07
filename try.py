import streamlit as st
import streamlit.components.v1 as components
import re
import os
import base64
import html
from io import BytesIO
from typing import Any, TypedDict

import requests
from PIL import Image, ImageOps
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

# Optional: lets iPhone HEIC/HEIF photos open too (pip/uv add pillow-heif). Safe if missing.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# WooCommerce Credentials & URL
STORE_URL = os.getenv("WC_STORE_URL", "https://pk.nexgentrend.com")
CK = os.getenv("WC_CONSUMER_KEY")
CS = os.getenv("WC_CONSUMER_SECRET")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL")
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")


class AgentInput(TypedDict):
    messages: list[AnyMessage | dict[str, Any]]


st.set_page_config(page_title="Nexgen Assistant", page_icon="🛒")

# Keep the currently visible warm palette alongside the original dark palettes.
THEMES = {
    "Warm Sand — Current": {
        "background": "linear-gradient(135deg, #fbf6ef 0%, #f1e5d5 100%)",
        "surface": "#fffaf4",
        "surface_alt": "#f4e9dc",
        "text": "#2a211c",
        "muted": "#75685d",
        "accent": "#e07b39",
        "highlight": "#f0c14b",
        "border": "#e6d5c2",
        "welcome": "linear-gradient(135deg, #1b1512, #2a211c)",
        "dark": False,
    },
    "Midnight Navy — Gradient": {
        "background": "linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)",
        "surface": "#1e293b",
        "surface_alt": "#29374d",
        "text": "#f8fafc",
        "muted": "#cbd5e1",
        "accent": "#f08a45",
        "highlight": "#f0c14b",
        "border": "#475569",
        "welcome": "linear-gradient(135deg, #1e293b, #334155)",
        "dark": True,
    },
    "Solid Dark — #121212": {
        "background": "#121212",
        "surface": "#1e1e1e",
        "surface_alt": "#292929",
        "text": "#f0f0f0",
        "muted": "#bdbdbd",
        "accent": "#e07b39",
        "highlight": "#f0c14b",
        "border": "#454545",
        "welcome": "linear-gradient(135deg, #1b1512, #2a211c)",
        "dark": True,
    },
}

with st.expander("🎨 Theme", expanded=False):
    theme_option = st.selectbox(
        "Color theme",
        list(THEMES),
        help="Choose a full app palette. Warm Sand matches the current brand colors.",
        key="theme_option",
    )
theme = THEMES[theme_option]


def _normalized_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def find_product_matches(query: str, products: dict[str, dict[str, Any]]) -> list[str]:
    """Return confidently matching product names, allowing punctuation and word-order variation."""
    normalized_query = _normalized_name(query)
    if not normalized_query:
        return []

    exact_matches = []
    scored_matches: list[tuple[float, str]] = []
    query_tokens = set(normalized_query.split())
    normalized_names = {
        name: _normalized_name(name)
        for name in products
    }
    full_name_matches = [
        name for name, normalized_name in normalized_names.items()
        if normalized_name == normalized_query
    ]
    if full_name_matches:
        return full_name_matches

    for name, normalized_name in normalized_names.items():
        if not normalized_name:
            continue
        if normalized_name == normalized_query:
            exact_matches.append(name)
            continue
        if len(normalized_name) >= 4 and normalized_name in normalized_query:
            exact_matches.append(name)
            continue
        name_tokens = set(normalized_name.split())
        overlap = len(name_tokens & query_tokens) / max(len(name_tokens), 1)
        if len(name_tokens) > 1 and overlap >= 0.6:
            scored_matches.append((overlap, name))

    if exact_matches:
        return exact_matches
    return [name for _, name in sorted(scored_matches, reverse=True)]

def _clean_product_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _normalize_content(item)
            if text:
                parts.append(text)
        return " ".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "message"):
            if key in content:
                text = _normalize_content(content[key])
                if text:
                    return text
        return str(content)
    return str(content)

def _to_langchain_message(message: dict[str, Any]) -> AnyMessage:
    role = str(message.get("role", "user")).lower()
    content = _normalize_content(message.get("content", ""))
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)

def fetch_products():
    """Fetch live products from WooCommerce REST API."""
    if not CK or not CS:
        return {}

    url = f"{STORE_URL}/wp-json/wc/v3/products"
    auth = (CK, CS)

    try:
        response = requests.get(url, auth=auth, timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"WooCommerce Fetch Error: {e}")
        return {}

    if not isinstance(data, list):
        return {}

    products = {}
    for item in data:
        if not isinstance(item, dict):
            continue

        name = _clean_product_name(item.get("name"))
        if not name:
            continue

        price = item.get("price", "0")
        currency = "PKR"

        stock_qty = item.get("stock_quantity")
        stock = stock_qty if stock_qty is not None else (item.get("stock_status") or "In Stock")

        rating = item.get("average_rating", "N/A")
        description = item.get("description", "")
        product_id = item.get("id")

        images = item.get("images", [])
        image_url = images[0].get("src") if isinstance(images, list) and images and isinstance(images[0], dict) else None

        products[name] = {
            "Price": price,
            "Currency": currency,
            "Stock": stock,
            "Rating": rating,
            "description": description,
            "variant_id": product_id,
            "image_url": image_url,
        }
    return products

@st.cache_data(ttl=300)
def get_products():
    return fetch_products()

@st.cache_data(ttl=600)
def _available_groq_models() -> list[str]:
    """Model IDs this GROQ_API_KEY can actually use (empty list if the lookup fails)."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return []
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))
    except Exception:
        return []


VISION_MODEL_CANDIDATES = [
    m for m in dict.fromkeys([
        VISION_MODEL,
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
    ]) if m
]


def prepare_image_for_vision(raw: bytes, max_side: int = 1280):
    """Open any image format Pillow understands and return (jpeg_bytes, 'image/jpeg'), or None."""
    try:
        img = Image.open(BytesIO(raw))
        img = ImageOps.exif_transpose(img)  # fix phone-photo rotation
        try:
            img.seek(0)  # first frame of GIF/WEBP animations
        except Exception:
            pass
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            img = background
        else:
            img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        out = BytesIO()
        img.save(out, format="JPEG", quality=88)
        return out.getvalue(), "image/jpeg"
    except Exception:
        return None


def describe_uploaded_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[str, str]:
    """Describe a product photo with a Groq vision model. Returns (description, error_message)."""
    if not image_bytes:
        return "", "The image was empty."

    available = _available_groq_models()
    candidates = [m for m in VISION_MODEL_CANDIDATES if not available or m in available]
    if not candidates:
        shown = ", ".join(available)[:700] or "could not load the model list"
        return "", (
            "None of the known vision models are available for this Groq key. "
            f"Set GROQ_VISION_MODEL to a vision-capable model from: {shown}"
        )

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Describe this product photo in 2-3 concise sentences: what type of "
                    "product it is, its color, material, and any distinguishing features. "
                    "This description will be used to match it against a store catalog."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
            },
        ]
    )

    last_error = ""
    for model_name in candidates:
        try:
            response = ChatGroq(model=model_name, temperature=0).invoke([message])
            text = _normalize_content(response.content).strip()
            if text:
                return text, ""
            last_error = f"{model_name} returned an empty description."
        except Exception as e:
            last_error = f"{model_name}: {e}"
    return "", last_error


@tool
def get_product(name: str) -> str:
    """Look up a product by name and return its price, stock, rating, description and image URL."""
    if not isinstance(name, str):
        name = str(name or "")

    products = get_products()
    if not products:
        return "No products are currently available or API credentials are missing."

    matches = find_product_matches(name, products)
    if not matches:
        return f"product not found. Available products: {', '.join(products.keys())}"
    if len(matches) > 1:
        return f"Several products matched. Please choose one: {', '.join(matches)}"
    return f"{matches[0]}: {products[matches[0]]}"

@tool
def list_products() -> str:
    """List all available products with their prices, stock, rating and image URL."""
    products = get_products()
    if not products:
        return "No products are currently available or API credentials are missing."

    lines = []
    for name, info in products.items():
        lines.append(
            f"- **{name}**: {info['Price']} {info['Currency']} | Stock: {info['Stock']} | "
            f"Rating: {info['Rating']} | Image: {info['image_url']}"
        )
    return "\n".join(lines)

@tool
def add_to_cart(product_name: str, quantity: int = 1) -> str:
    """Add a product to the cart by name and return a checkout link."""
    if not isinstance(product_name, str):
        product_name = str(product_name or "")

    cleaned_name = product_name.strip()
    if not cleaned_name:
        return "Please provide a valid product name."

    try:
        quantity = int(quantity)
    except (TypeError, ValueError):
        return "Quantity must be a number."

    if quantity <= 0:
        return "Quantity must be greater than zero."

    products = get_products()
    if not products:
        return "No products are currently available."

    matches = find_product_matches(cleaned_name, products)
    if not matches:
        return f"Product not found. Available products: {', '.join(products.keys())}"
    if len(matches) > 1:
        return f"Please choose a specific product: {', '.join(matches)}"
    p = products[matches[0]]
    if not p.get("variant_id"):
        return "Sorry, this product cannot be added to cart right now."

    product_id = p["variant_id"]
    checkout_url = f"{STORE_URL}/checkout/?add-to-cart={product_id}&quantity={quantity}"

    marker = f"[[SYNC_CART:{product_id}:{quantity}]]"
    return f"Added {quantity} x {cleaned_name} to cart! Complete your order here: {checkout_url} {marker}"

@st.cache_resource
def get_agent():
    llm = ChatGroq(model=CHAT_MODEL, temperature=0)
    return create_agent(
        llm,
        tools=[get_product, list_products, add_to_cart],
        system_prompt=(
            "You are a helpful product assistant for an online tech store (NexgenTrend). "
            "Reply in the language used by the user. Use only catalog facts returned by tools; "
            "never invent availability, prices, ratings, or product matches. "
            "For a catalog request, call list_products. For a product question, call get_product "
            "with the best-matching name without asking for confirmation first. "
            "For a purchase request, call add_to_cart and include its checkout link. "
            "Keep replies concise and readable: use a plain list for one attribute, and a small "
            "Markdown table only when comparing multiple attributes across products. "
            "Do not add columns or details the user did not ask for."
        ),
    )

agent = get_agent()

# Apply the selected palette consistently across Streamlit's native surfaces.
st.markdown(f"""
<style>
    html, body {{
        background: {theme["background"]} !important;
        background-attachment: fixed !important;
    }}
    #MainMenu, footer, header {{visibility: hidden;}}
    div[class*="viewerBadge"] {{ display: none !important; }}
    a[href*="streamlit.io"] {{ display: none !important; }}
    .block-container {{max-width: 1100px; padding-top: 1.5rem; padding-bottom: 5rem;}}
    .stApp, [data-testid="stAppViewContainer"] {{
        background: {theme["background"]} !important;
        background-attachment: fixed !important;
        color: {theme["text"]} !important;
    }}
    section[data-testid="stAppScrollToBottomContainer"] {{
        background: {theme["background"]} !important;
        background-attachment: fixed !important;
    }}
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {{
        background: {theme["background"]} !important;
        background-color: {theme["surface"]} !important;
        border-top: 1px solid {theme["border"]};
    }}
    [data-testid="stBottom"] > div {{
        background: {theme["background"]} !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background: {theme["surface"]} !important;
        border-right: 1px solid {theme["border"]};
    }}
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
    [data-testid="stFileUploader"] label, [data-testid="stChatMessage"] {{
        color: {theme["text"]} !important;
    }}
    [data-testid="stChatMessage"] {{
        background: {theme["surface"]};
        border: 1px solid {theme["border"]};
        border-radius: 16px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
    }}
    [data-testid="stChatInput"], [data-testid="stFileUploader"] section,
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {theme["surface"]} !important;
        border-color: {theme["border"]} !important;
        border-radius: 16px !important;
    }}
    [data-testid="stChatInput"] > div {{
        background: {theme["surface_alt"]} !important;
        border-radius: 14px !important;
    }}
    [data-testid="stChatInput"] textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] input,
    [data-testid="stSelectbox"] button {{
        color: {theme["text"]} !important;
        background-color: {theme["surface"]} !important;
        border-color: {theme["border"]} !important;
    }}
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background: {theme["surface"]} !important;
        color: {theme["text"]} !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {theme["muted"]} !important;
        opacity: 1;
    }}
    [data-testid="stChatInput"] button, [data-testid="stLinkButton"] a {{
        color: #ffffff !important;
        background: {theme["accent"]} !important;
        border-color: {theme["accent"]} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stChatInput"] button:hover,
    [data-testid="stLinkButton"] a:hover {{
        filter: brightness(1.08);
    }}
    [data-testid="stFileUploader"] button {{
        color: {theme["text"]} !important;
        background: {theme["surface_alt"]} !important;
        border-color: {theme["accent"]} !important;
    }}
    [data-testid="stFileUploader"] section {{
        border: 1px dashed {theme["accent"]} !important;
        padding: 1rem;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        transition: transform 160ms ease, box-shadow 160ms ease;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.12);
    }}
    .welcome-box {{
        background: {theme["welcome"]};
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        margin: 0.5rem 0 1.25rem;
        text-align: center;
        border: 1px solid {theme["border"]};
        border-bottom: 4px solid {theme["accent"]};
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.14);
    }}
    .welcome-box h3 {{
        margin: 0 0 0.4rem;
        font-size: 1.5rem;
        color: {theme["highlight"]};
    }}
    .welcome-box p {{
        margin: 0;
        color: #ffffff;
        font-size: 0.95rem;
        opacity: 0.92;
    }}
    @media (max-width: 640px) {{
        .block-container {{padding: 1rem 0.75rem 5rem;}}
        .welcome-box {{padding: 1.1rem;}}
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="welcome-box">
    <h3>🛒 Nexgen Assistant</h3>
    <p>Ask about products, upload a photo, or compare prices and availability.</p>
</div>
""", unsafe_allow_html=True)


def render_product_cards(product_names: list[str], products: dict[str, dict[str, Any]]) -> None:
    """Render matching WooCommerce products in responsive cards."""
    if not product_names:
        return
    columns = st.columns(min(3, len(product_names)))
    for index, name in enumerate(product_names):
        info = products.get(name)
        if not info:
            continue
        with columns[index % len(columns)]:
            with st.container(border=True):
                image_url = info.get("image_url")
                if image_url:
                    st.image(image_url, use_container_width=True)
                else:
                    st.caption("Product image unavailable")
                st.markdown(f"**{html.escape(name)}**")
                price = info.get("Price") or "Contact for price"
                currency = info.get("Currency", "PKR")
                st.markdown(f"### {html.escape(str(price))} {html.escape(str(currency))}")
                stock = info.get("Stock", "Availability unknown")
                st.caption(f"Availability: {stock}  ·  Rating: {info.get('Rating', 'N/A')}")
                product_id = info.get("variant_id")
                if product_id:
                    st.link_button(
                        "Add to cart",
                        f"{STORE_URL}/checkout/?add-to-cart={product_id}&quantity=1",
                        use_container_width=True,
                    )


if "history" not in st.session_state:
    st.session_state.history = []
if "last_processed_photo" not in st.session_state:
    st.session_state.last_processed_photo = None

for msg in st.session_state.history:
    avatar = "🧑" if msg["role"] == "user" else "🛒"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("products"):
            render_product_cards(msg["products"], get_products())

def handle_query(user_message: str):
    """Send a message to the agent and render the reply with safe error fallback."""
    if not user_message or not user_message.strip():
        return

    st.session_state.history.append({"role": "user", "content": user_message})

    try:
        with st.spinner("Thinking..."):
            message_history: list[AnyMessage | dict[str, Any]] = []
            for message in st.session_state.history:
                message_history.append(_to_langchain_message(message))
            agent_input: AgentInput = {"messages": message_history}
            result = agent.invoke(agent_input)
            
            # Safe extraction of response from various agent return formats
            if isinstance(result, dict) and "messages" in result:
                last_msg = result["messages"][-1]
                reply = getattr(last_msg, "content", str(last_msg))
            else:
                reply = str(result)
    except Exception:
        if re.search(r"\b(all|list|show|catalog|catalogue)\b", user_message.lower()):
            prods = get_products()
            if prods:
                reply = "Here are the products currently available in the catalog."
            else:
                reply = "I couldn't load the catalog just now. Please try again shortly."
        else:
            reply = "I couldn't process that request just now. Please try again."

    reply = _normalize_content(reply)
    products = get_products()
    normalized_query = _normalized_name(user_message)
    is_catalog_request = bool(
        re.search(r"\b(catalog|catalogue)\b", normalized_query)
        or re.search(r"\b(all|every|list|show)\b.*\b(products?|items?)\b", normalized_query)
        or re.search(r"\b(products?|items?)\b.*\b(list|catalog|catalogue)\b", normalized_query)
    )
    mentioned_products = (
        list(products)
        if is_catalog_request
        else find_product_matches(f"{user_message}\n{reply}", products)
    )
    st.session_state.history.append(
        {"role": "assistant", "content": reply, "products": mentioned_products}
    )

    match = re.search(r"\[\[SYNC_CART:(\d+):(\d+)\]\]", reply)
    display_reply = re.sub(r"\[\[SYNC_CART:\d+:\d+\]\]", "", reply).strip()

    with st.chat_message("assistant", avatar="🛒"):
        st.markdown(display_reply)
        render_product_cards(mentioned_products, products)
        if match:
            product_id, qty = match.group(1), match.group(2)
            components.html(f"""
                <script>
                window.top.postMessage({{
                    type: 'ADD_TO_WOOCOMMERCE_CART',
                    productId: '{product_id}',
                    quantity: {qty}
                }}, '*');
                </script>
            """, height=0)

# --- Photo-based product search ---
uploaded_photo = st.file_uploader(
    "📷 Search Products By Photo",
    type=None,  # accept every file type; real images are validated with Pillow below
    key="product_photo_uploader",
    help="Any image format works: JPG, PNG, WEBP, GIF, BMP, TIFF, HEIC, AVIF and more.",
)

if uploaded_photo is not None:
    photo_signature = f"{uploaded_photo.name}:{uploaded_photo.size}"
    if st.session_state.last_processed_photo != photo_signature:
        st.session_state.last_processed_photo = photo_signature
        prepared = prepare_image_for_vision(uploaded_photo.getvalue())

        if prepared is None:
            st.warning(
                "This file could not be read as an image. Please upload a photo "
                "(JPG, PNG, WEBP, GIF, BMP, TIFF, HEIC, AVIF...)."
            )
        else:
            image_bytes, mime_type = prepared

            with st.chat_message("user", avatar="🧑"):
                st.image(image_bytes, width=200)
                st.caption("Uploaded photo")

            with st.spinner("Analyzing the photo..."):
                description, vision_error = describe_uploaded_image(image_bytes, mime_type)

            if description:
                photo_query = (
                    "I uploaded a photo of a product. Description of the photo: "
                    f"{description}\n\nPlease find the best-matching product(s) in our catalog "
                    "and tell me about their availability, price, and rating."
                )
                handle_query(photo_query)
            else:
                st.warning("Photo search is unavailable right now. Please try text search.")
                if vision_error:
                    with st.expander("Technical details"):
                        st.code(vision_error[:900])

# --- Text-based chat ---
question = st.chat_input("Ask about our products...")
if question:
    with st.chat_message("user", avatar="🧑"):
        st.write(question)
    handle_query(question)