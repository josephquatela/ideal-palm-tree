I worked independently on this project, taking full ownership from problem framing through to the final presentation.

### What I learned about the challenge

The core insight I didn't anticipate going in was that accuracy and usefulness are not the same thing. My models never achieved low MAPE — the magnitude of predicted outcomes was often unreliable. But directional accuracy reached 93.7% on the best model, meaning the system correctly tells a founder whether an impact will be positive or negative almost all of the time. For a decision-support tool, that's actually the more actionable signal. A founder comparing two scenario branches doesn't need to know their inventory drops exactly 40 units — they need to know which scenario is safer and roughly how bad the downside looks. Reframing success around that distinction changed how I evaluated everything downstream.

I also learned that the ceiling on a machine learning system is often a data problem, not a model problem. The three hardest branch types — supplier_change, net_terms_change, contract_change — remained above 200% MAPE regardless of what architecture or regularisation strategy I applied. Running a grouped model experiment specifically to test this was one of the more valuable things I did: it produced a meaningful negative result and let me write with confidence that the path forward requires new features, not more tuning.

### Design for an augmented intelligence challenge

I designed the system to produce confidence-ranged intervals rather than point estimates from the start, treating honesty about uncertainty as a design requirement rather than a bonus feature. The conformal calibration layer made that mathematically rigorous — the system guarantees 80% coverage, which is something you can explain to a non-technical founder and actually stand behind. The "what ships today" framing in the final report was a deliberate design decision: shipping directional signals and calibrated intervals for price and promo scenarios while being explicit about what doesn't work yet.

### Feature engineering

The most consequential feature engineering decisions were: exponential decay weighting on rolling windows (recent commits matter more), per-business revenue normalisation to make retail and B2B comparable across a 40× scale difference, and replacing a single polymorphic magnitude column with nine type-specific magnitude features. That last one directly improved Ridge regression performance by giving the model separable coefficients per scenario type.

### Select, audit and improve algorithms

I followed a structured progression: baseline → XGBoost → Transformer → audit → six targeted experiments. The baseline report was the pivotal step — it surfaced three specific, fixable problems rather than a vague sense that results were bad. Each experiment targeted one diagnosed issue, was isolated in its own reproducible notebook, and ended with a pass/fail check against hard-coded baseline numbers.

### Interact with stakeholders

The final deliverable was designed for a product audience, not an ML one. I built a 16-slide deck that leads with the business problem, explains results in terms of directional accuracy and confidence ranges, names what is ready to ship and why, and is honest about where the system fails and what would actually fix it.