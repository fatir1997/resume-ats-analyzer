def call_gemini_with_retry(
    client,
    model_name: str,
    prompt: str
):

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            response = client.models.generate_content(

                model=model_name,

                contents=prompt,

                config={
                    "temperature": 0.2,

                    # Tell Gemini to return JSON
                    "response_mime_type": "application/json",

                    # Force the exact JSON structure
                    "response_schema": ATS_RESPONSE_SCHEMA,
                },
            )

            return response

        except Exception as exc:

            last_error = exc

            error_text = str(exc).lower()

            is_temporary = (
                "503" in error_text
                or
                "unavailable" in error_text
                or
                "high demand" in error_text
                or
                "429" in error_text
                or
                "resource exhausted" in error_text
            )

            if not is_temporary:

                raise

            if attempt < MAX_RETRIES - 1:

                wait_seconds = 2 ** attempt

                time.sleep(
                    wait_seconds
                )

    raise RuntimeError(
        "Gemini API is temporarily unavailable "
        "after multiple retry attempts."
        f"\n\nLast error: {last_error}"
    )
