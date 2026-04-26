ANTICIPATED JUDGE QUESTIONS

Q: "How do you handle no ground truth?"
A: "Three approaches. Internal consistency — does equipment match claims. Rule-based prerequisites — surgery requires anesthesia. External web verification via Tavily. Plus confidence intervals that widen when data is sparse. We built self-correction into every layer."

Q: "Why not Databricks?"
A: "Our architecture maps directly: ChromaDB to Mosaic AI Vector Search, OpenAI to Agent Bricks, our pipeline to Genie Code, MLflow for observability. We designed for portability — in production this runs natively on Databricks with Unity Catalog."

Q: "How much did this cost?"
A: "Under one dollar for all 10,000 facilities. GPT-4o-mini with intelligent batching, automatic retry, and a sanitizer that auto-corrects formatting errors. 99.95% success rate."

Q: "Show me a facility your system caught."
A: "Krishna Homeopathy Research Hospital, Jaipur. LLM extraction flagged it claiming surgery and emergency — suspicious for homeopathy. Trust scoring penalized missing surgical equipment. Tavily confirmed: zero surgical capabilities online. Three layers, same conclusion."

Q: "What about confidence scoring?"
A: "Every trust score has a prediction interval. Width is 30 minus completeness times 0.25. Sparse data means wider intervals. We're honest about uncertainty — exactly what your Areas of Research section asked about."

Q: "How does the agent avoid hallucinating?"
A: "Five-step pipeline. The Validator actively rejects results that don't match. Tavily cross-references the real web. MLflow logs every step for auditability. If verification fails, the agent says so transparently."

Q: "Social impact?"
A: "1,745 districts across 33 states. We found districts with 500,000 people and zero trustworthy dialysis. An NGO sees exactly where to deploy a mobile dialysis unit. That's reducing Discovery-to-Care time."

Q: "What would you improve?"
A: "Full MLflow 3 integration on Databricks, real-time pipeline updates, expanding web verification to all 10,000 facilities, and Unity Catalog governance for data lineage."

Q: "Can you try a live query?"
[Use backup queries:]
- "Best cardiac care in Hyderabad"
- "Hospitals with ICU in Bihar"
- "Oncology treatment in Tamil Nadu"
- "Hospitals claiming surgery but no anesthesiologist"

---

FINAL FRONTEND CHECKLIST — test each one:

1. Click "Suspicious dental clinics in India"
   □ Reasoning steps appear in sidebar (should show 5 steps including Web Verification)
   □ Map pins drop with correct colors
   □ Clicking a pin opens right drawer with real trust scores

2. Click "Worst dialysis deserts in India"
   □ Desert heatmap renders with real districts
   □ Red zones visible
   □ Clicking a district shows desert score and top gaps

3. Click "Emergency C-section in rural Maharashtra"
   □ Results appear on map
   □ Chain of thought mentions location filtering

4. Click any pin → Right drawer shows:
   □ Trust score ring with number
   □ 4 sub-score progress bars
   □ Capabilities with colored dots (green/blue/red/gray)
   □ Contradictions in red section
   □ Confidence interval visible

5. Type a custom query and press Enter
   □ Works without crashing
