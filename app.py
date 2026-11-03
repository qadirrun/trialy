
import streamlit as st
from rapidfuzz import process, fuzz
import re

# Page configuration
st.set_page_config(
    page_title="Equipment Name Matcher",
    layout="centered"
)

st.title("Equipment Name Matcher")
st.caption("Find the 3 closest alternative names using fuzzy matching.")

st.divider()

# Equipment name
equipment = st.text_input(
    "Equipment Name",
    placeholder="e.g. Analysis Time"
)

# Alternative names
alternatives = st.text_area(
    "Alternative Names",
    placeholder="quoted delivery time | timer | autonomy time | stress analysis required | life cycle analysis",
    height=200
)


# Matching function
def find_best_matches(equipment, alternatives, limit=3):

    if not equipment.strip() or not alternatives.strip():
        return []

    # Split alternatives using | or newline
    alternative_list = [
        name.strip()
        for name in re.split(r"[|\n]+", alternatives)
        if name.strip()
    ]

    if not alternative_list:
        return []

    # Remove duplicate alternatives
    alternative_list = list(dict.fromkeys(alternative_list))

    # Find top 3 matches
    results = process.extract(
        equipment.strip(),
        alternative_list,
        scorer=fuzz.WRatio,
        limit=limit
    )

    return [
        {
            "rank": index + 1,
            "name": name,
            "score": round(score, 2)
        }
        for index, (name, score, choice_index) in enumerate(results)
    ]


# Matching button
if st.button("Find Top 3 Matches", type="primary", use_container_width=True):

    if not equipment.strip():
        st.warning("Please enter an equipment name.")

    elif not alternatives.strip():
        st.warning("Please enter alternative names.")

    else:

        results = find_best_matches(equipment, alternatives)

        if results:

            st.divider()
            st.subheader("Top 3 Matching Results")

            for result in results:

                st.markdown(f"**Rank {result['rank']}**")

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.text_input(
                        f"Alternative Name {result['rank']}",
                        value=result["name"],
                        disabled=True,
                        key=f"name_{result['rank']}"
                    )

                with col2:
                    st.metric(
                        "Score",
                        f"{result['score']}%"
                    )

                st.divider()

            # Final output: all 3 nearest names
            final_names = " | ".join(
                result["name"] for result in results
            )

            st.subheader("Final Output")

            st.text_area(
                "final",
                value=final_names,
                disabled=True
            )

            st.download_button(
                "Download Results (TXT)",
                data=final_names,
                file_name="equipment_matches.txt",
                mime="text/plain",
                use_container_width=True
            )

        else:
            st.error("No matching alternatives found.")
