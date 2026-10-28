"""
AI-powered resume analysis using Groq API (LLaMA model).

Provides detailed resume feedback, strengths/weaknesses, and improvement suggestions.
"""
import json
import os

from groq import Groq

GROQ_API_KEY = os.environ.get(
    'GROQ_API_KEY',
    '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'
)

client = Groq(api_key=GROQ_API_KEY)


def analyze_resume(resume_text, predicted_category, jd_text=None):
    """Send the resume to Groq LLM for detailed analysis and suggestions.

    Returns a dict with keys:
        summary, strengths, weaknesses, suggestions, skills_found,
        missing_skills, overall_rating, experience_level, ats_tips
    """
    jd_section = ""
    if jd_text:
        jd_section = f"""
The candidate applied for a role with this job description:
---
{jd_text[:2000]}
---
Also evaluate how well the resume matches this specific job description.
Include a 'jd_match_feedback' key with 2-3 sentences on how well the resume
aligns with the JD and what to add/remove to improve the match.
"""

    prompt = f"""You are an expert resume reviewer and career counselor. Analyze the following resume
that has been classified into the "{predicted_category}" job category.

Resume text:
---
{resume_text[:4000]}
---
{jd_section}

Provide a thorough analysis in the following JSON format (respond ONLY with valid JSON, no markdown):
{{
    "summary": "A 2-3 sentence professional summary of the candidate based on their resume",
    "experience_level": "Entry Level / Mid Level / Senior Level / Executive",
    "overall_rating": <number from 1 to 10>,
    "strengths": ["strength 1", "strength 2", "strength 3"],
    "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
    "skills_found": ["skill1", "skill2", "skill3"],
    "missing_skills": ["Recommended skill 1 for {predicted_category}", "skill 2", "skill 3"],
    "suggestions": [
        "Specific actionable suggestion 1 to improve the resume",
        "Specific actionable suggestion 2",
        "Specific actionable suggestion 3",
        "Specific actionable suggestion 4",
        "Specific actionable suggestion 5"
    ],
    "ats_tips": [
        "ATS optimization tip 1",
        "ATS optimization tip 2",
        "ATS optimization tip 3"
    ]{', "jd_match_feedback": "..."' if jd_text else ''}
}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional resume analyst. Always respond with valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_completion_tokens=1500,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        analysis = json.loads(raw)
        return analysis

    except json.JSONDecodeError:
        return _fallback_analysis(predicted_category)
    except Exception as e:
        print(f"Groq API error: {e}")
        return _fallback_analysis(predicted_category)


def _fallback_analysis(category):
    """Return a generic analysis when the API call fails."""
    return {
        "summary": "Unable to generate AI analysis at this time. Please try again later.",
        "experience_level": "Unknown",
        "overall_rating": 0,
        "strengths": ["Could not analyze - API unavailable"],
        "weaknesses": ["Could not analyze - API unavailable"],
        "skills_found": [],
        "missing_skills": [],
        "suggestions": [
            "Ensure your resume includes a clear professional summary",
            "List relevant technical skills for the " + category + " field",
            "Quantify achievements with numbers and metrics",
            "Keep the resume concise (1-2 pages)",
            "Tailor the resume to each job application"
        ],
        "ats_tips": [
            "Use standard section headings (Experience, Education, Skills)",
            "Avoid tables, images, and complex formatting",
            "Include keywords from the job description"
        ],
        "api_error": True
    }
