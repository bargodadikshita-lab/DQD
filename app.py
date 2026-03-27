import streamlit as st
from main import DuplicateDetector
from gemini_ai import generate_answer
import os

st.set_page_config(
    page_title="DQD: AI-Based Semantic Duplicate Question Detection and Answering System",
    layout="centered"
)

# Load detector
detector = DuplicateDetector("data/qa_dataset.csv")

# Smaller Title
st.markdown(
    "<h3 style='text-align: center;'>DQD: AI-Based Semantic Duplicate Question Detection and Answering System</h3>",
    unsafe_allow_html=True
)

user_input = st.text_input("Enter your question:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a question")
    else:
        result = detector.find_similar(user_input)

        score = result["best_score"]

        st.write("Similarity Score:", round(score, 2))

        # STRONG MATCH
        if score > 0.80:
            st.success("Best Match Found")

            best = result["matches"][0]

            st.write("### Best Match")
            st.write("Question:", best["question"])
            st.write("Answer:", best["answer"])

            # Similar Questions
            st.write("### Similar Questions")

            for match in result["matches"][1:]:
                with st.expander(
                    f"{match['question']} (Score: {round(match['score'], 2)})"
                ):
                    st.write("Answer:", match["answer"])

        # WEAK MATCH
        elif score > 0.65:
            st.warning("Weak Matches Found")

            for match in result["matches"]:
                with st.expander(
                    f"{match['question']} (Score: {round(match['score'], 2)})"
                ):
                    st.write("Answer:", match["answer"])

        # NO MATCH → GEMINI
        else:
            st.error("No relevant match found")

            st.write("Generating AI Answer...")
            ai_answer = generate_answer(user_input)

            st.write("### AI Answer")
            st.write(ai_answer)

            # SAVE + RELOAD
            if st.button("Save this Q&A"):
                detector.add_new_qa(user_input, ai_answer, "data/qa_dataset.csv")
                st.success("Saved successfully")

                if os.path.exists("data/embeddings.pt"):
                    os.remove("data/embeddings.pt")

                detector = DuplicateDetector("data/qa_dataset.csv")

                st.info("System updated with new knowledge. Please search again.")