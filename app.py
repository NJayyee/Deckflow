# app.py
import os
import json
from pathlib import Path

import streamlit as st
import pdfplumber
from pptx import Presentation
from openai import OpenAI

# -------- config --------
MODEL = "gpt-4o-mini"
OUTPUT_SCHEMA_HINT = {
    "emails": [{"subject": "string", "body": "string", "cta": "string"}],
    "linkedin": ["string"],
    "script": "string",
}

# -------- helpers --------
def load_css(path: str) -> None:
    try:
        css = Path(path).read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def extract_pdf_text(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt.strip())
    return "\n\n---\n\n".join(parts)

def extract_pptx_text(path: str) -> str:
    prs = Presentation(path)
    slides = []
    for slide in prs.slides:
        items = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                t = (shape.text or "").strip()
                if t:
                    items.append(t)
        if items:
            slides.append("\n".join(items))
    return "\n\n---\n\n".join(slides)

def to_outline(raw: str) -> dict:
    chunks = [s.strip() for s in raw.split("\n\n---\n\n") if s.strip()]
    if not chunks:
        chunks = [raw.strip()]
    slides = [{"index": i + 1, "text": s} for i, s in enumerate(chunks[:60])]
    return {"slides": slides}

def openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY environment variable")
        st.stop()
    return OpenAI(api_key=api_key)

def chat_json(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def coerce_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1 :].strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start : end + 1])
        except Exception:
            st.error("Model returned invalid JSON. Try again or simplify style notes.")
            st.stop()
    st.error("No JSON found in model response.")
    st.stop()

def plan_brief(client: OpenAI, outline: dict, persona: str, tone: str, style: str) -> dict:
    system = "You extract concise marketing briefs for outbound work. Return pure JSON."
    user = (
        f"Persona: {persona}\n"
        f"Tone: {tone}\n"
        f"Style notes: {style}\n"
        f"Deck outline JSON:\n{json.dumps(outline, ensure_ascii=False)}\n\n"
        "Return JSON with fields: "
        "audience (string), pains (array of strings), value (array of strings), "
        "proofs (array of strings, only from the deck), objections (array of strings)."
    )
    return coerce_json(chat_json(client, system, user))

def generate_assets(client: OpenAI, brief: dict) -> dict:
    system = (
        "You are a senior B2B marketer. "
        "Output pure JSON matching this schema:\n"
        f"{json.dumps(OUTPUT_SCHEMA_HINT)}"
    )
    rules = (
        "Write a 5 step outbound email sequence, 3 LinkedIn posts, and a 90 second talk track.\n"
        "Rules:\n"
        "- One CTA per email.\n"
        "- Subject under 50 characters.\n"
        "- Email body under 140 words.\n"
        "- Plain language.\n"
        "- Every claim must map to a proof in brief.proofs.\n"
        "Return JSON only with keys: emails, linkedin, script."
    )
    user = f"Brief JSON:\n{json.dumps(brief, ensure_ascii=False)}\n\n{rules}"
    return coerce_json(chat_json(client, system, user))

def guardrail_review(client: OpenAI, draft: dict, style: str, brief: dict) -> dict:
    system = (
        "You review for clarity, tone, and proof alignment. "
        "Keep the same JSON keys and structure. Return pure JSON."
    )
    user = (
        f"Draft JSON:\n{json.dumps(draft, ensure_ascii=False)}\n\n"
        f"Brand notes: {style}\n"
        f"Allowed proofs:\n{json.dumps(brief.get('proofs', []), ensure_ascii=False)}\n\n"
        "Rewrite or remove any claim not supported by an allowed proof. "
        "Reading level grade 7 to 9. Keep one CTA per email. "
        "Do not add keys. Return JSON only."
    )
    return coerce_json(chat_json(client, system, user))

def validate_output(data: dict) -> list[str]:
    errors = []
    emails = data.get("emails", [])
    if not isinstance(emails, list) or not emails:
        errors.append("emails missing")
    for i, e in enumerate(emails, 1):
        subj = (e.get("subject") or "").strip()
        body = (e.get("body") or "").strip()
        cta = (e.get("cta") or "").strip()
        if len(subj) > 50:
            errors.append(f"email {i} subject too long")
        if not cta:
            errors.append(f"email {i} missing CTA")
        if len(body.split()) > 160:
            errors.append(f"email {i} body too long")
    if not data.get("linkedin"):
        errors.append("linkedin posts missing")
    if not (isinstance(data.get("script"), str) and data.get("script").strip()):
        errors.append("script missing")
    return errors

def header() -> None:
    st.markdown(
        """
<div class="bf-header">
  <div class="bf-left">
    <span class="bf-logo-badge"></span><span>Brainfish</span>
  </div>
  <div class="bf-links">
    <a href="#">Why Brainfish</a>
    <a href="#">Product</a>
    <a href="#">Solutions</a>
    <a href="#">Resources</a>
    <a href="#">Pricing</a>
  </div>
  <div class="bf-right">
    <button class="bf-btn">Sign in</button>
    <button class="bf-btn cta">Book Demo</button>
  </div>
</div>

<div class="bf-hero">
  

  <div class="bf-chip">Ambient AI Agent for Marketing Teams</div>
  <h1 class="bf-h1">Turn Sales Presentations into Outbound Sequences</h1>
  <p class="bf-sub">Upload a deck, get emails, LinkedIn posts, and a talk track.</p>
  <div class="btns">
    <a class="btn primary" href="#agent">Try Now</a>
    <a class="btn secondary" href="#">Learn More</a>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

def footer() -> None:
    st.markdown(
        """
<div class="bf-footer">
  <p>© 2025 Brainfish Mock | Demo only</p>
  <p><a href="#">Docs</a> | <a href="#">Blog</a> | <a href="#">Careers</a></p>
</div>
""",
        unsafe_allow_html=True,
    )

# -------- app --------
def main() -> None:
    st.set_page_config(page_title="Deck → Outbound Agent", page_icon="📨", layout="centered")
    load_css("styles.css")
    header()

    # Step 1
    st.markdown('<div id="agent"></div>', unsafe_allow_html=True)
    with st.container():
        st.header("Step 1. Upload your sales deck")
        uploaded = st.file_uploader("PDF or PPTX", type=["pdf", "pptx"])

    # Step 2
    with st.container():
        st.header("Step 2. Set persona and voice")
        col1, col2 = st.columns(2)
        with col1:
            persona = st.text_input("Target persona", "Head of Operations")
            tone = st.selectbox("Tone", ["Professional", "Friendly", "Direct", "Technical"])
        with col2:
            style = st.text_area("Brand voice notes", "Clear, concise, value focused.")
        run = st.button("Generate", type="primary", use_container_width=True)

    if not run:
        footer()
        return
    if not uploaded:
        st.warning("Upload a deck first.")
        footer()
        return

    # Step 3
    with st.container():
        st.header("Step 3. We read your deck")
        tmp_path = Path(f"./_uploads/{uploaded.name}")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.read())

        with st.spinner("Extracting slides"):
            raw = extract_pdf_text(str(tmp_path)) if uploaded.name.lower().endswith(".pdf") else extract_pptx_text(str(tmp_path))
            if not raw.strip():
                st.error("No text found in the deck. Add speaker notes or try another file.")
                footer()
                return
            outline = to_outline(raw)
        with st.expander("Preview extracted outline", expanded=False):
            st.code(json.dumps(outline, ensure_ascii=False, indent=2), language="json")

    # Step 4
    with st.container():
        st.header("Step 4. Draft your outbound pack")
        client = openai_client()

        with st.spinner("Planning brief"):
            brief = plan_brief(client, outline, persona, tone, style)

        with st.expander("Brief summary", expanded=True):
            st.write(f"Audience: {brief.get('audience','')}")
            pains = brief.get("pains", [])
            if pains:
                st.write("Pains:")
                for p in pains:
                    st.write(f"- {p}")
            proofs = brief.get("proofs", [])
            if proofs:
                st.write("Top proofs:")
                for p in proofs[:3]:
                    st.write(f"- {p}")

        with st.spinner("Generating drafts"):
            draft = generate_assets(client, brief)

        with st.spinner("Applying guardrails"):
            final = guardrail_review(client, draft, style, brief)

        issues = validate_output(final)
        if issues:
            st.info("Quality checks")
            for e in issues:
                st.write(f"- {e}")

    # Step 5
    with st.container():
        st.header("Results")
        tabs = st.tabs(["Emails", "LinkedIn", "Talk track", "JSON"])
        emails = final.get("emails", [])
        linkedin = final.get("linkedin", [])
        script = final.get("script", "")

        with tabs[0]:
            for i, e in enumerate(emails, 1):
                st.markdown(f"**Step {i}: {e.get('subject','')}**")
                st.write(e.get("body", ""))
                st.write(f"CTA: {e.get('cta','')}")
                st.divider()

        with tabs[1]:
            for i, p in enumerate(linkedin, 1):
                st.markdown(f"**Post {i}**")
                st.write(p)
                st.divider()

        with tabs[2]:
            st.markdown("**Talk track**")
            st.write(script)

        with tabs[3]:
            st.code(json.dumps(final, ensure_ascii=False, indent=2), language="json")

        st.download_button(
            "Download JSON",
            data=json.dumps(final, ensure_ascii=False, indent=2),
            file_name="outbound.json",
            mime="application/json",
            use_container_width=True,
        )

    footer()

if __name__ == "__main__":
    main()
