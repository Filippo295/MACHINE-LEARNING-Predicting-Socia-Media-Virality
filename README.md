# Predicting Socia Media Virality

A full data science pipeline built in collaboration with Uniting Group to analyze what drives engagement in sponsored Instagram and TikTok content, from data cleaning and LLM-based feature extraction to statistical modeling and a multi-agent system that replicates the entire workflow on new data.

---

## The Problem

Brands pay creators to post sponsored videos, but knowing what actually makes those videos work is not straightforward. This project studies sponsored content on Instagram and TikTok and asks a direct question: what drives engagement, and can a brand predict it or steer it? If the right content features, the right creators, and the right timing can be identified, a brand can plan campaigns better and spend its budget where it pays off.

---

## Data & Methodology

**Dataset:** 3,243 sponsored videos provided by Uniting Group, covering Instagram Stories, Instagram Reels, and TikTok posts. Variables span performance metrics (reach, likes, comments, link clicks), creator information (followers, gender), brand information (industry, Italian vs international, brand follower count), and low-level video measures computed from frames (motion, saturation, luminance, contrast, sharpness, color complexity).

**Pipeline:**

1. **Data Cleaning:** Missing values for 51 Reels/TikTok posts were imputed by matching each post's percentile in the comments distribution to the same percentile in the likes distribution. Missing clicks for 267 Stories were filled using each creator's average clicks-per-reach multiplied by the post's reach.
2. **Manual Feature Engineering:** A 6-category time slot variable (night, morning, lunch, afternoon, dinner, evening) was derived from posting time. Month and a weekend flag (Friday to Sunday) were extracted from the post date. A binary face variable was added to flag whether the creator's face appears in the video.
3. **Target Definition:** The dataset was split by content type. For Reels and TikToks, the main target is Engagement Rate defined as (Likes + 20 x Comments) / Reach, with comments weighted 20x because they represent a higher-effort action. A secondary target, Comments per Like (Comments / Likes), measures how much a post sparks conversation versus passive approval. For Stories, the target is Click-Through Rate (Clicks / Reach).
4. **LLM Feature Extraction:** Most of the content signal lives in the video and caption, not in the numeric columns. Gemini 2.5 Flash was used to extract structured variables from every video and caption, chosen for its strong video understanding, and run via the Gemini API on Google Cloud. Three separate extraction sessions were run: one on the full video for visual and audio features, one on the first three seconds for hook strength, and one on the caption text. Each prompt assigned the model a role as an expert social media marketing analyst, described every variable category in depth with examples, and forced output into a fixed JSON schema (Pydantic) with a closed list of allowed values and temperature set to zero.

## Extracted features

**Video-based variables**
- Tone: enthusiastic/energetic, ironic/playful, calm/confidential, logical/informative, neutral/detached
- Voice speed: slow, normal, fast, absent
- Microkinetics (based on a model in literature using Arousal and Dominance dimensions): charismatic_performer, authoritative_expert, soft_engager, intimate_confidant, absent
- Presence of activity: spoken only, spoken with activities, activities only, neither spoken nor activities
- Format: tutorial, basic placement, brand-related experience/event, slice of life, product review
- Product integration: in use, on display, on stage, absent
- Funnel: awareness, consideration, conversion
- Product positioning: affordable, premium, luxury, unidentifiable

**Caption-based variables**
- Tone: enthusiastic/energetic, sarcastic, calm/confidential, logical/informative, neutral/detached
- Funnel: awareness, consideration, conversion
- Caption length: calculated from text

**First 3 seconds**
- Hook strength: strong, medium, weak (categorical, after finding that a continuous scale clustered around a single point and lost discriminatory power)
---

## Models

**Clustering:** The goal was to identify which creators deliver the most engagement relative to their audience size. Each creator was summarized to a single row using the median of their posts to avoid viral outliers distorting the picture. K-means (k=3, chosen by silhouette score) grouped creators into three tiers: Macro, Mid-size, and Micro. A Pareto frontier was then drawn to identify which creators no other creator beats on both followers and engagement simultaneously.

**Regression:** The goal was to measure what actually drives engagement up or down. Lasso dropped irrelevant features first, then OLS was fitted on the survivors. The previous post's performance was added as a lag variable to account for within-creator correlation. Separate models were run for Reels, Stories, and the Macro and Micro clusters within Stories.

**Markov Chains:** The goal was to test whether posts have momentum. Each post was labeled as 1 (above that creator's own median reach) or 0 (below), and a transition matrix measured how sticky good and bad periods are.

**Moran's I:** The goal was to test whether strong posts cluster together in time or appear at random. The index treats the 3 posts before and after each post as neighbors and measures whether high-reach posts sit next to other high-reach posts. It complements the Markov analysis by widening the view from one post ahead to a window of six, allowing to identify cold and hot periods of the creators.

**Hidden Markov Models:** The goal was to detect hidden performance regimes without imposing any threshold manually, and to measure how sponsorship performs inside each. Unlike the Markov chain, HMM lets the data find the regimes on its own. Run on an extended dataset that also included organic (non-sponsored) posts.

**Classification:** The goal was to understand what separates a post that sparks comments from one that only collects likes. Comments per Like was cut (based on the plot of its distribution) at the 75th percentile into Silent and Conversational classes. A Random Forest ranked features by SHAP importance, then simpler models were re-trained on the top 5/10/15/20/25/30 features and evaluated on macro F1, with the train/validation gap monitored to avoid overfitting, allowing to identify the best set of features available.

---

**Clustering:** Macro and Micro creators were roughly equal in terms of performance, but Mid-size creators outperformed both significantly on engagement rate for Reels and on CTR for Stories, with the Pareto frontier confirming this pattern. The sweet spot is mid-size: big enough to reach a real audience, not so big that engagement thins out.

**Regression:** The strongest predictor across all models was the previous post's performance: the first lever is not the content itself but the choice of the creator. For Reels, rich visuals, long posts, and strong hooks help, while posts that feel like an ad drag it down. For Stories, pushing directly toward conversion and keeping visuals clean drives clicks, with smaller creators outperforming bigger ones on CTR. Micro creators work best with a spontaneous style like a friend giving an advice, while Macro creators need a sober message aimed directly and clearly at conversion.

**Markov Chains:** For Reels, a weak post had a 71% probability of being followed by another weak one, while a strong one had only 58% of being followed by another strong one. For Stories, low states stayed low 69 to 86% of the time, high states only 55 to 73%. This means that posts have momemntum and in particular cold periods are stickier than hot ones. It is thus important to detect hot periods in advance and ride them, while acting fast to break cold ones.

**Moran's I:** Three creators showed positive autocorrelation (indices from 0.118 to 0.284), meaning their strong posts come in streaks. Others instead showed a negative index (-0.218), meaning their peaks are isolated and thus harder to predict. This means that it is important to time branded content around hot streaks for the first group in order to maximize performance, while treating the second unpredictable group post by post. 

**Hidden Markov Models:** The same sponsored post received approximately 9 times the views in a hot state versus a cold one. On Instagram, sponsored posts received 26 to 62% fewer views than organic in ordinary periods, a structural cost brands should price in. TikTok peaks also shrink over time while Instagram stays stable, making Instagram more predictable for planning. The takeaway is to choose the creator and the moment before worrying about anything else. Past performance is the single biggest predictor of the next post, so book creators in a good run ride their hot streaks.

**Classification:** Naive Bayes on the top 5 SHAP features achieved a macro F1 of 0.638 with minimal overfitting. The five features were reach, followers, face frame ratio, caption length, and motion level. Larger audiences, more face on screen, longer captions, and faster motion all pushed posts toward Silent. If the goal is discussion and not just passive likes, pick smaller creators with loyal communities and leave viewers something to react to. When a video and its caption already say everything, people watch and move on instead of commenting.

---

## The Multi-Agent System

The final part of the project replicated the entire pipeline as a multi-agent system built in Claude Code, so the same analysis can run on a new dataset with no manual steps.

The system is organized into 15 single-purpose skills covering data cleaning, feature engineering, LLM extraction (full video, first three seconds, caption), one for each of the six statistical models, exploratory analysis, and report generation. All skills read from a single shared configuration file so column names, thresholds, and paths live in one place.

Seven roles run the pipeline as a team: Project Manager, Data Engineer, Feature Extractor, Statistical Modeller, Exploratory Analyst, Quality Reviewer, and Programmer.

On a full run over approximately 6,000 sponsored reels, the agent reproduced the project's main findings. The previous post's performance coefficient was again the dominant predictor (+0.50). One limitation stood out: the agent cannot be given full autonomy on judgment calls like model selection, where it got stuck in a loop and required human intervention.

**Cost and time:** The full-video Gemini extraction accounted for roughly 85% of both runtime and cost, estimated at around $33 for the full run.

---

## Tech Stack

**LLM Extraction**
- Gemini 2.5 Flash via the Gemini API on Google Cloud: video, hook, and caption feature extraction with structured JSON output

**Statistical Modeling**
- Python: Lasso, OLS, K-means, Random Forest, Naive Bayes, Hidden Markov Models (hmmlearn), Markov transition matrices
- SHAP: feature importance ranking

**Data & Analysis**
- Pandas, NumPy: data manipulation and feature engineering
- Matplotlib, Seaborn: EDA plots, confusion matrices, training curves
- scikit-learn: model training and evaluation

**Validation App**
- Claude Code (vibe-coded): internal tool for comparing LLM labels against manual annotations

**Agent System**
- Claude Code: multi-agent pipeline orchestration
