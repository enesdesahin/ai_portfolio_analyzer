import streamlit as st
from typing import Tuple

GITHUB_PROFILE_URL = None
LINKEDIN_PROFILE_URL = None

_SOCIAL_SVGS = {
    "github": (
        "GitHub",
        """<svg viewBox="0 0 24 24" role="img" aria-label="GitHub" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.726-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.757-1.333-1.757-1.089-.745.083-.73.083-.73 1.205.085 1.84 1.236 1.84 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.418-1.305.762-1.605-2.665-.304-5.466-1.332-5.466-5.932 0-1.31.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23a11.5 11.5 0 0 1 3.003-.404 c1.018.005 2.045.138 3.003.404 2.291-1.552 3.297-1.23 3.297-1.23.655 1.653.244 2.874.12 3.176.77.84 1.235 1.911 1.235 3.221 0 4.61-2.807 5.625-5.479 5.921.43.372.823 1.102.823 2.222 0 1.604-.015 2.896-.015 3.286 0 .319.216.694.825.576 C20.565 22.092 24 17.592 24 12.297 24 5.67 18.627.297 12 .297z"/></svg>""",
    ),
    "linkedin": (
        "LinkedIn",
        """<svg viewBox="0 0 448 512" role="img" aria-label="LinkedIn" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M416 32H32A32 32 0 0 0 0 64v384a32 32 0 0 0 32 32h384a32 32 0 0 0 32-32V64a32 32 0 0 0-32-32zM135.4 416H69.1V202.2h66.3zm-33.1-243a38.4 38.4 0 1 1 38.4-38.4 38.4 38.4 0 0 1-38.4 38.4zM384 416h-66.2V312c0-24.8-.5-56.7-34.5-56.7-34.5 0-39.8 27-39.8 54.9V416h-66.2V202.2h63.6v29.2h.9c8.9-16.8 30.6-34.5 63-34.5 67.3 0 79.7 44.4 79.7 102.1z"/></svg>""",
    ),
}

def get_social_links_html(
    *,
    github_url: str | None = None,
    linkedin_url: str | None = None,
    clean_layout: bool = False,
    show_attribution: bool = True,
) -> str | None:
    """Generate the HTML for social profile badges."""
    github_target = github_url or GITHUB_PROFILE_URL
    linkedin_target = linkedin_url or LINKEDIN_PROFILE_URL

    entries = []
    if github_target:
        entries.append(("github", github_target))
    if linkedin_target:
        entries.append(("linkedin", linkedin_target))

    if not entries:
        return None

    badges = []
    for key, url in entries:
        label, svg_markup = _SOCIAL_SVGS.get(key, ("", ""))
        if not svg_markup:
            continue
        extra_style = "margin-left:-8px;" if key == "linkedin" else ""
        badge = (
            "<a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\" "
            "style=\"display:inline-flex;width:38px;height:38px;border-radius:999px;"
            "align-items:center;justify-content:center;color:#202124;text-decoration:none;{extra_style}\" "
            "title=\"{label}\">"
            "<span style=\"display:inline-flex;width:22px;height:22px;color:#202124;\">{svg}</span>"
            "</a>"
        ).format(url=url, label=label, svg=svg_markup, extra_style=extra_style)
        badges.append(badge)

    if not badges:
        return None

    # Construct the inner content
    attribution_html = ""
    if show_attribution:
        attribution_html = "<div style=\"margin-top:0.45rem;font-size:0.8rem;color:rgba(32,33,36,0.65);\">Developed by Enes SAHIN</div>"
    
    inner_html = (
        "<div style=\"display:flex;flex-direction:column;align-items:flex-start;\">"
        "<div style=\"font-size:0.85rem;font-weight:400;color:#202124;margin-bottom:0.4rem;\">Connect with me</div>"
        f"<div style=\"display:flex;gap:0.3rem;margin-left:-6px;\">{''.join(badges)}</div>"
        f"{attribution_html}"
        "</div>"
    )
    return inner_html

def render_social_links(
    *,
    github_url: str | None = None,
    linkedin_url: str | None = None,
    clean_layout: bool = False,
) -> None:
    """Render social profile badges at the bottom of the sidebar."""
    content_html = get_social_links_html(github_url=github_url, linkedin_url=linkedin_url, clean_layout=clean_layout)
    # Valid content check
    if not content_html:
        return

    # Add divider only for standard layout (Builder/Analytics)
    if not clean_layout:
        st.sidebar.divider()

    # Render as a styled card for ALL pages (matching Home page style)
    # Background: rgba(255, 255, 255, 0.04), Border: 1px solid rgba(255, 255, 255, 0.1), Radius: 8px
    card_style = (
        "background-color: rgba(0, 0, 0, 0.04);"
        "border: 1px solid rgba(0, 0, 0, 0.1);"
        "border-radius: 8px;"
        "padding: 16px;"
        "margin-bottom: 20px;"
    )
    
    card_html = f'<div style="{card_style}">{content_html}</div>'
    st.sidebar.markdown(card_html, unsafe_allow_html=True)
