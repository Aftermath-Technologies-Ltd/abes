# Author: Bradley R. Kinnard
"""
1000-prompt cognitive evaluation bank — parameterized templates across 8 domains.

Each prompt carries a gold_answer for semantic cosine scoring (no keywords),
ecology_checks for internal belief-state auditing, and session_group for
multi-turn coherence. Templates are varied with different surface content
to avoid overfitting to any single phrasing.

Domains (125 each): episodic_memory, semantic_memory, working_memory,
selective_attention, language_comprehension, reasoning, social_cognition,
self_correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalPrompt:
    """Single evaluation prompt with gold answer for semantic scoring."""
    id: int
    domain: str
    construct: str
    message: str
    gold_answer: str
    forbidden_semantics: list[str] = field(default_factory=list)
    session_group: str = ""
    reference: str = ""
    ecology_checks: list[str] = field(default_factory=list)
    is_setup: bool = False
    horizon: str = "immediate"


def build_prompts() -> list[EvalPrompt]:
    """Build and return exactly 1000 evaluation prompts."""
    prompts: list[EvalPrompt] = []
    _counter = [0]  # mutable so nested add() can increment

    def add(domain: str, construct: str, message: str, gold_answer: str = "",
            *, ref: str = "", group: str = "", forbidden: list[str] | None = None,
            checks: list[str] | None = None, setup: bool = False,
            horizon: str = "immediate"):
        _counter[0] += 1
        prompts.append(EvalPrompt(
            id=_counter[0], domain=domain, construct=construct,
            message=message, gold_answer=gold_answer,
            forbidden_semantics=forbidden or [], session_group=group,
            reference=ref, ecology_checks=checks or [],
            is_setup=setup, horizon=horizon,
        ))

    _episodic_memory(add)
    _semantic_memory(add)
    _working_memory(add)
    _selective_attention(add)
    _language_comprehension(add)
    _reasoning(add)
    _social_cognition(add)
    _self_correction(add)

    return prompts


# ====================================================================
# DOMAIN 1: EPISODIC MEMORY (125 prompts)
# Tulving 1972 — encoding, temporal/spatial retrieval, ordering, source
# ====================================================================

def _episodic_memory(add):
    dom, ref = "episodic_memory", "Tulving 1972"

    # --- encoding (25 prompts: 13 setup + 12 probes) ---
    facts = [
        ("I had a job interview at Google last Monday.", "ep_enc_1",
         "When was my job interview?", "Your job interview at Google was last Monday."),
        ("I adopted a rescue cat named Whiskers two weeks ago.", "ep_enc_2",
         "What pet did I recently adopt?", "You adopted a rescue cat named Whiskers two weeks ago."),
        ("My sister graduated from nursing school in December.", "ep_enc_3",
         "When did my sister graduate?", "Your sister graduated from nursing school in December."),
        ("I started taking piano lessons every Wednesday evening.", "ep_enc_4",
         "What lessons am I taking?", "You are taking piano lessons every Wednesday evening."),
        ("Last summer I visited the Grand Canyon with my family.", "ep_enc_5",
         "Where did I go last summer?", "You visited the Grand Canyon with your family last summer."),
        ("I got promoted to senior engineer in January.", "ep_enc_6",
         "What happened in January?", "You got promoted to senior engineer in January."),
        ("My best friend moved to Portland last month.", "ep_enc_7",
         "Where did my best friend move?", "Your best friend moved to Portland last month."),
        ("I ran my first half marathon in October.", "ep_enc_8",
         "What did I do in October?", "You ran your first half marathon in October."),
        ("We renovated the kitchen and added a breakfast island.", "ep_enc_9",
         "What did we do to the kitchen?", "You renovated the kitchen and added a breakfast island."),
        ("I switched from iPhone to Samsung last week.", "ep_enc_10",
         "What phone switch did I make?", "You switched from iPhone to Samsung last week."),
        ("My daughter won the science fair with her volcano project.", "ep_enc_11",
         "What did my daughter do at the science fair?", "Your daughter won the science fair with her volcano project."),
        ("I started volunteering at the food bank on Saturdays.", "ep_enc_12",
         "Where do I volunteer?", "You volunteer at the food bank on Saturdays."),
    ]
    # emit setup+probe pair, plus one extra setup for count
    add(dom, "encoding", "I finished reading War and Peace over the holidays.",
        ref=ref, group="ep_enc_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold in facts:
        add(dom, "encoding", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "encoding", probe_msg, gold, ref=ref, group=grp)

    # --- temporal_retrieval (25 prompts: 13 setup + 12 probes) ---
    times = [
        ("My dentist appointment was last Tuesday at 3pm.", "ep_time_1",
         "When was my dentist appointment?", "Your dentist appointment was last Tuesday at 3pm."),
        ("I submitted the quarterly report on Friday morning.", "ep_time_2",
         "When did I submit the quarterly report?", "You submitted the quarterly report on Friday morning."),
        ("The power went out at our house Wednesday night.", "ep_time_3",
         "When did the power go out?", "The power went out at your house Wednesday night."),
        ("I signed the lease for the new apartment on March 1st.", "ep_time_4",
         "When did I sign the lease?", "You signed the lease for the new apartment on March 1st."),
        ("My flight to Chicago departs at 6am on Thursday.", "ep_time_5",
         "When does my Chicago flight depart?", "Your flight to Chicago departs at 6am on Thursday."),
        ("I had my annual physical on the 15th of last month.", "ep_time_6",
         "When was my annual physical?", "Your annual physical was on the 15th of last month."),
        ("The team standup moved to 9:30am starting Monday.", "ep_time_7",
         "When is the team standup now?", "The team standup moved to 9:30am starting Monday."),
        ("I booked the hotel for our anniversary on June 10th.", "ep_time_8",
         "When is our anniversary hotel booked for?", "Your anniversary hotel is booked for June 10th."),
        ("The contractor is coming to fix the roof on Saturday at 8am.", "ep_time_9",
         "When is the contractor coming?", "The contractor is coming to fix the roof on Saturday at 8am."),
        ("My car inspection expires on April 30th.", "ep_time_10",
         "When does my car inspection expire?", "Your car inspection expires on April 30th."),
        ("I have a parent-teacher conference at 4pm next Wednesday.", "ep_time_11",
         "When is the parent-teacher conference?", "Your parent-teacher conference is at 4pm next Wednesday."),
        ("The company all-hands meeting is scheduled for the 25th.", "ep_time_12",
         "When is the company all-hands meeting?", "The company all-hands meeting is scheduled for the 25th."),
    ]
    add(dom, "temporal_retrieval", "My tax filing deadline is April 15th.",
        ref=ref, group="ep_time_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold in times:
        add(dom, "temporal_retrieval", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "temporal_retrieval", probe_msg, gold, ref=ref, group=grp)

    # --- spatial_retrieval (25 prompts: 13 setup + 12 probes) ---
    places = [
        ("I left my laptop charger at the coffee shop on Main Street.", "ep_loc_1",
         "Where did I leave my laptop charger?", "You left your laptop charger at the coffee shop on Main Street."),
        ("The new Italian restaurant is on 5th Avenue next to the bookstore.", "ep_loc_2",
         "Where is the new Italian restaurant?", "The new Italian restaurant is on 5th Avenue next to the bookstore."),
        ("I parked my car on level 3, section B of the parking garage.", "ep_loc_3",
         "Where did I park my car?", "You parked your car on level 3, section B of the parking garage."),
        ("My gym membership is at the Fitness Center on Oak Boulevard.", "ep_loc_4",
         "Where is my gym?", "Your gym is the Fitness Center on Oak Boulevard."),
        ("The spare house key is hidden under the blue flower pot on the porch.", "ep_loc_5",
         "Where is the spare house key?", "The spare house key is hidden under the blue flower pot on the porch."),
        ("I keep my passport in the top drawer of the filing cabinet.", "ep_loc_6",
         "Where do I keep my passport?", "You keep your passport in the top drawer of the filing cabinet."),
        ("The kids' soccer practice is at Riverside Park, field 4.", "ep_loc_7",
         "Where is the kids' soccer practice?", "The kids' soccer practice is at Riverside Park, field 4."),
        ("My favorite hiking trail starts at the Elk Meadow trailhead.", "ep_loc_8",
         "Where does my favorite hiking trail start?", "Your favorite hiking trail starts at the Elk Meadow trailhead."),
        ("The vet clinic I use is on Maple Drive across from the school.", "ep_loc_9",
         "Where is my vet clinic?", "Your vet clinic is on Maple Drive across from the school."),
        ("I stashed emergency cash in the zippered pocket of my blue backpack.", "ep_loc_10",
         "Where is my emergency cash?", "Your emergency cash is in the zippered pocket of your blue backpack."),
        ("Our wedding venue is the Rose Garden at Lakewood Estate.", "ep_loc_11",
         "Where is our wedding venue?", "Your wedding venue is the Rose Garden at Lakewood Estate."),
        ("The bike repair shop I go to is on 12th Street near the bridge.", "ep_loc_12",
         "Where is the bike repair shop?", "The bike repair shop is on 12th Street near the bridge."),
    ]
    add(dom, "spatial_retrieval", "My storage unit is at SafeKeep on Industrial Road.",
        ref=ref, group="ep_loc_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold in places:
        add(dom, "spatial_retrieval", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "spatial_retrieval", probe_msg, gold, ref=ref, group=grp)

    # --- temporal_ordering (25 prompts: 10 setup + 15 probes) ---
    # 5 sequences of 2 setup + 3 probes each
    sequences = [
        ("ep_ord_1",
         ["First I went to the bank, then I stopped at the pharmacy, then I picked up groceries.",
          "After groceries I went home and cooked dinner."],
         [("What did I do right after the bank?", "You stopped at the pharmacy after the bank."),
          ("What was the last thing I did?", "The last thing you did was cook dinner at home."),
          ("Did I go to the pharmacy before or after groceries?", "You went to the pharmacy before getting groceries.")]),
        ("ep_ord_2",
         ["Monday I had a meeting, Tuesday I wrote the report, Wednesday I presented it.",
          "Thursday I got feedback, Friday I revised the report."],
         [("What did I do the day after writing the report?", "You presented the report on Wednesday, the day after writing it on Tuesday."),
          ("Which came first, the feedback or the presentation?", "The presentation came first on Wednesday, then feedback on Thursday."),
          ("What was the last thing I did that week?", "The last thing you did was revise the report on Friday.")]),
        ("ep_ord_3",
         ["In the morning I went jogging, at noon I had lunch with Sarah, in the afternoon I worked on the garden.",
          "In the evening I watched a movie and then read before bed."],
         [("What did I do after lunch with Sarah?", "You worked on the garden in the afternoon after lunch with Sarah."),
          ("What was my last activity of the day?", "Your last activity was reading before bed."),
          ("Did I jog before or after lunch?", "You went jogging in the morning before having lunch at noon.")]),
        ("ep_ord_4",
         ["Step 1 was mixing the ingredients, step 2 was kneading the dough, step 3 was letting it rise.",
          "Step 4 was shaping the loaves, step 5 was baking at 375 degrees."],
         [("What came after kneading the dough?", "After kneading the dough you let it rise in step 3."),
          ("What was the final step?", "The final step was baking at 375 degrees."),
          ("What did I do before shaping the loaves?", "Before shaping the loaves you let the dough rise.")]),
        ("ep_ord_5",
         ["I flew from Denver to Dallas, then from Dallas to Miami.",
          "From Miami I took a cruise to the Bahamas, then flew back to Denver."],
         [("What city did I visit between Denver and Miami?", "You stopped in Dallas between Denver and Miami."),
          ("Where did I go after Miami?", "After Miami you took a cruise to the Bahamas."),
          ("What was my final destination?", "Your final destination was Denver, flying back from the Bahamas.")]),
    ]
    for grp, setups, probes in sequences:
        for s in setups:
            add(dom, "temporal_ordering", s, ref=ref, group=grp,
                setup=True, checks=["belief_created"])
        for probe_msg, gold in probes:
            add(dom, "temporal_ordering", probe_msg, gold, ref=ref, group=grp)

    # --- source_monitoring (25 prompts: 10 setup + 15 probes) ---
    # 5 scenarios: attribute info to different sources, then ask who said what
    sources = [
        ("ep_src_1",
         ["My doctor told me to cut back on sodium.",
          "My nutritionist said I should eat more leafy greens."],
         [("Who told me to reduce sodium?", "Your doctor told you to cut back on sodium."),
          ("What did my nutritionist recommend?", "Your nutritionist said you should eat more leafy greens."),
          ("Did my doctor say anything about leafy greens?", "No, it was your nutritionist who recommended leafy greens, not your doctor.")]),
        ("ep_src_2",
         ["My boss said the project deadline is next Friday.",
          "My coworker mentioned the client wants a demo on Wednesday."],
         [("Who told me about the deadline?", "Your boss said the project deadline is next Friday."),
          ("What did my coworker say?", "Your coworker mentioned the client wants a demo on Wednesday."),
          ("Did my boss mention the demo?", "No, it was your coworker who mentioned the demo, not your boss.")]),
        ("ep_src_3",
         ["I read in the newspaper that the local bridge will close for repairs.",
          "My neighbor told me the construction will take three months."],
         [("Where did I learn about the bridge closure?", "You read about the bridge closure in the newspaper."),
          ("Who said the construction will take three months?", "Your neighbor told you the construction will take three months."),
          ("Did the newspaper mention the duration?", "No, the newspaper mentioned the bridge closure but it was your neighbor who told you about the three-month duration.")]),
        ("ep_src_4",
         ["My wife reminded me that her parents are visiting next weekend.",
          "My son told me he needs new cleats for soccer."],
         [("Who told me about the visit from the in-laws?", "Your wife reminded you that her parents are visiting next weekend."),
          ("What does my son need?", "Your son told you he needs new cleats for soccer."),
          ("Did my wife mention soccer cleats?", "No, it was your son who mentioned needing soccer cleats, not your wife.")]),
        ("ep_src_5",
         ["The mechanic said my brake pads need replacing within a month.",
          "The dealership quoted me $800 for a full brake service."],
         [("Who warned me about the brake pads?", "The mechanic said your brake pads need replacing within a month."),
          ("How much did the dealership quote?", "The dealership quoted you $800 for a full brake service."),
          ("Did the mechanic give me a price?", "No, the mechanic warned about the brake pads but it was the dealership that quoted the $800 price.")]),
    ]
    for grp, setups, probes in sources:
        for s in setups:
            add(dom, "source_monitoring", s, ref=ref, group=grp,
                setup=True, checks=["belief_created"])
        for probe_msg, gold in probes:
            add(dom, "source_monitoring", probe_msg, gold, ref=ref, group=grp)


# ====================================================================
# DOMAIN 2: SEMANTIC MEMORY (125 prompts)
# Collins & Quillian 1969 — facts, categories, properties, discrimination
# ====================================================================

def _semantic_memory(add):
    dom, ref = "semantic_memory", "Collins & Quillian 1969"

    # --- fact_encoding (25 prompts: 13 setup + 12 probes) ---
    facts = [
        ("My favorite programming language is Rust.", "sm_fact_1",
         "What is my favorite programming language?", "Your favorite programming language is Rust."),
        ("I'm allergic to shellfish.", "sm_fact_2",
         "What food allergy do I have?", "You are allergic to shellfish."),
        ("My car is a 2019 Honda Civic in silver.", "sm_fact_3",
         "What car do I drive?", "You drive a 2019 Honda Civic in silver."),
        ("I have two kids, a boy aged 7 and a girl aged 4.", "sm_fact_4",
         "How many kids do I have?", "You have two kids, a boy aged 7 and a girl aged 4."),
        ("My blood type is O negative.", "sm_fact_5",
         "What is my blood type?", "Your blood type is O negative."),
        ("I work as a data scientist at a fintech startup.", "sm_fact_6",
         "What do I do for work?", "You work as a data scientist at a fintech startup."),
        ("My wife's name is Rebecca and she's a veterinarian.", "sm_fact_7",
         "What is my wife's name and job?", "Your wife's name is Rebecca and she is a veterinarian."),
        ("I take 20mg of lisinopril daily for blood pressure.", "sm_fact_8",
         "What medication do I take?", "You take 20mg of lisinopril daily for blood pressure."),
        ("My mortgage payment is $2,400 per month.", "sm_fact_9",
         "How much is my mortgage payment?", "Your mortgage payment is $2,400 per month."),
        ("I studied mechanical engineering at Georgia Tech.", "sm_fact_10",
         "Where did I go to college?", "You studied mechanical engineering at Georgia Tech."),
        ("My dog is a golden retriever named Biscuit.", "sm_fact_11",
         "What kind of dog do I have?", "You have a golden retriever named Biscuit."),
        ("I was born in Portland, Oregon in 1988.", "sm_fact_12",
         "Where was I born?", "You were born in Portland, Oregon in 1988."),
    ]
    add(dom, "fact_encoding", "My social security number ends in 4982.",
        ref=ref, group="sm_fact_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold in facts:
        add(dom, "fact_encoding", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "fact_encoding", probe_msg, gold, ref=ref, group=grp)

    # --- categorical_inference (25 prompts: 13 setup + 12 probes) ---
    # "X is a Y. Ys have property Z. Does X have Z?"
    categories = [
        ("My new pet Ziggy is a parrot.", "sm_cat_1",
         "Can my pet Ziggy fly?", "Since Ziggy is a parrot and parrots can fly, yes Ziggy can likely fly."),
        ("I just bought a Tesla Model 3.", "sm_cat_2",
         "Does my new car use gasoline?", "No, since your Tesla Model 3 is an electric vehicle, it does not use gasoline."),
        ("My nephew plays the cello.", "sm_cat_3",
         "Is the cello a string instrument?", "Yes, the cello is a string instrument that is played with a bow."),
        ("I'm growing basil in my herb garden.", "sm_cat_4",
         "Is my basil plant edible?", "Yes, basil is an edible herb commonly used in cooking."),
        ("My friend just got a Bernese Mountain Dog.", "sm_cat_5",
         "Is a Bernese Mountain Dog a large breed?", "Yes, Bernese Mountain Dogs are a large breed, typically weighing 70-115 pounds."),
        ("I collect vinyl records.", "sm_cat_6",
         "Are vinyl records an analog format?", "Yes, vinyl records are an analog audio format that stores sound as physical grooves."),
        ("My grandmother just learned to use a tablet computer.", "sm_cat_7",
         "Does a tablet have a touchscreen?", "Yes, tablets have touchscreens as their primary input method."),
        ("We planted a Japanese maple in the front yard.", "sm_cat_8",
         "Will our Japanese maple lose its leaves in winter?", "Yes, Japanese maples are deciduous trees that lose their leaves in autumn and winter."),
        ("I started brewing kombucha at home.", "sm_cat_9",
         "Is kombucha a fermented drink?", "Yes, kombucha is a fermented tea drink made using a SCOBY culture."),
        ("My daughter is learning to play the oboe.", "sm_cat_10",
         "Is the oboe a woodwind instrument?", "Yes, the oboe is a woodwind instrument played with a double reed."),
        ("I just installed solar panels on my roof.", "sm_cat_11",
         "Do solar panels generate electricity from sunlight?", "Yes, solar panels convert sunlight into electricity using photovoltaic cells."),
        ("My coworker drives a Harley-Davidson.", "sm_cat_12",
         "Is a Harley-Davidson a motorcycle?", "Yes, Harley-Davidson is a motorcycle manufacturer known for heavyweight cruiser bikes."),
    ]
    add(dom, "categorical_inference", "I'm learning to play the banjo.",
        ref=ref, group="sm_cat_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold in categories:
        add(dom, "categorical_inference", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "categorical_inference", probe_msg, gold, ref=ref, group=grp)

    # --- property_retrieval (25 prompts: 13 setup + 12 probes) ---
    properties = [
        ("My house has 4 bedrooms, 2.5 bathrooms, and a finished basement.", "sm_prop_1",
         "How many bathrooms does my house have?", "Your house has 2.5 bathrooms."),
        ("My laptop is a MacBook Pro with 32GB RAM and an M3 chip.", "sm_prop_2",
         "How much RAM does my laptop have?", "Your laptop has 32GB of RAM."),
        ("I drive a silver 2019 Honda Civic with a manual transmission.", "sm_prop_3",
         "What transmission does my car have?", "Your car has a manual transmission."),
        ("My desk is a standing desk made of walnut with an electric motor.", "sm_prop_4",
         "What type of wood is my desk made of?", "Your standing desk is made of walnut."),
        ("My guitar is a 1972 Gibson Les Paul in sunburst finish.", "sm_prop_5",
         "What year is my guitar from?", "Your Gibson Les Paul guitar is from 1972."),
        ("My running shoes are Nike Pegasus 40 in size 11.", "sm_prop_6",
         "What size are my running shoes?", "Your running shoes are size 11."),
        ("My phone plan is with T-Mobile, unlimited data, $75 per month.", "sm_prop_7",
         "How much is my phone plan?", "Your phone plan with T-Mobile costs $75 per month."),
        ("My apartment is on the 8th floor, unit 812, with a balcony facing east.", "sm_prop_8",
         "Which direction does my balcony face?", "Your balcony faces east."),
        ("My watch is a Seiko Presage with a blue dial and leather strap.", "sm_prop_9",
         "What color is my watch dial?", "Your Seiko Presage watch has a blue dial."),
        ("The backyard deck is 20 feet by 12 feet, made of composite material.", "sm_prop_10",
         "How big is my deck?", "Your deck is 20 feet by 12 feet."),
        ("My coffee maker is a Breville Barista Express with a built-in grinder.", "sm_prop_11",
         "Does my coffee maker have a grinder?", "Yes, your Breville Barista Express has a built-in grinder."),
        ("I have a 65-inch Samsung QLED TV mounted in the living room.", "sm_prop_12",
         "How big is my TV?", "Your Samsung QLED TV is 65 inches."),
    ]
    add(dom, "property_retrieval", "My bike is a Specialized Roubaix, carbon frame, size 56.",
        ref=ref, group="sm_prop_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold in properties:
        add(dom, "property_retrieval", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "property_retrieval", probe_msg, gold, ref=ref, group=grp)

    # --- source_discrimination (25 prompts: 13 setup + 12 probes) ---
    # user facts vs general knowledge — system should not confuse them
    discriminations = [
        ("I believe the best programming language is Python.", "sm_disc_1",
         "What is the most popular programming language?",
         "The most popular programming language by many rankings is Python or JavaScript, depending on the metric.",
         ["Your favorite programming language is Python"]),
        ("My favorite planet is Saturn because of its rings.", "sm_disc_2",
         "What is the largest planet in our solar system?",
         "The largest planet in our solar system is Jupiter.",
         ["Your favorite planet is Saturn"]),
        ("I think pizza is the best food ever invented.", "sm_disc_3",
         "What country did pizza originate from?",
         "Pizza originated from Italy, specifically Naples.",
         ["Your favorite food is pizza"]),
        ("My favorite historical figure is Abraham Lincoln.", "sm_disc_4",
         "Who was the first president of the United States?",
         "The first president of the United States was George Washington.",
         ["Your favorite historical figure is Abraham Lincoln"]),
        ("I always say that cats are better than dogs.", "sm_disc_5",
         "What is the most popular pet in America?",
         "Dogs are the most popular pet in America by household ownership.",
         ["You think cats are better"]),
        ("I believe that electric cars are the future.", "sm_disc_6",
         "When was the first gasoline automobile invented?",
         "The first gasoline-powered automobile was invented by Karl Benz in 1886.",
         ["You believe electric cars are the future"]),
        ("My opinion is that summer is the best season.", "sm_disc_7",
         "Which season has the shortest days in the northern hemisphere?",
         "Winter has the shortest days in the northern hemisphere.",
         ["Your favorite season is summer"]),
        ("I think the Beatles are the greatest band of all time.", "sm_disc_8",
         "What year did the Beatles break up?",
         "The Beatles broke up in 1970.",
         ["Your favorite band is the Beatles"]),
        ("My favorite city to visit is Tokyo.", "sm_disc_9",
         "What is the most populous city in the world?",
         "Tokyo is the most populous metropolitan area in the world.",
         []),  # no forbidden — Tokyo is actually the answer here
        ("I believe meditation is the key to mental health.", "sm_disc_10",
         "What is cognitive behavioral therapy?",
         "Cognitive behavioral therapy (CBT) is a type of psychotherapy that helps people change negative thought patterns.",
         ["You believe meditation is important"]),
        ("My favorite number is 42.", "sm_disc_11",
         "What is the value of pi to two decimal places?",
         "The value of pi to two decimal places is 3.14.",
         ["Your favorite number is 42"]),
        ("I think renewable energy is more important than space exploration.", "sm_disc_12",
         "When did humans first land on the moon?",
         "Humans first landed on the moon on July 20, 1969 during the Apollo 11 mission.",
         ["You think renewable energy is more important"]),
    ]
    add(dom, "source_discrimination", "I believe the best movie ever made is The Shawshank Redemption.",
        ref=ref, group="sm_disc_0", setup=True, checks=["belief_created"])
    for item in discriminations:
        setup_msg, grp, probe_msg, gold = item[0], item[1], item[2], item[3]
        forbidden = item[4] if len(item) > 4 else []
        add(dom, "source_discrimination", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "source_discrimination", probe_msg, gold, ref=ref, group=grp,
            forbidden=forbidden)

    # --- knowledge_update (25 prompts: 10 setup + 15 probes) ---
    # Tell fact, then correct it, then verify the correction stuck
    updates = [
        ("sm_upd_1",
         ["My phone number is 555-0123.", "Actually, I changed my phone number. It's now 555-9876."],
         [("What is my phone number?", "Your phone number is 555-9876.",
           ["Your phone number is 555-0123"])]),
        ("sm_upd_2",
         ["I live in Seattle.", "We just moved. I now live in Austin."],
         [("Where do I live?", "You live in Austin.",
           ["You live in Seattle"])]),
        ("sm_upd_3",
         ["My salary is $95,000.", "I got a raise. My salary is now $110,000."],
         [("What is my salary?", "Your salary is $110,000.",
           ["Your salary is $95,000"])]),
        ("sm_upd_4",
         ["I work at Microsoft.", "I left Microsoft. I now work at Stripe."],
         [("Where do I work?", "You work at Stripe.",
           ["You work at Microsoft"])]),
        ("sm_upd_5",
         ["My car is a blue Toyota Camry.", "I sold the Camry. I now drive a red Mazda 3."],
         [("What car do I drive?", "You drive a red Mazda 3.",
           ["You drive a blue Toyota Camry"])]),
        ("sm_upd_6",
         ["My daughter goes to Lincoln Elementary.", "She transferred. She now goes to Washington Middle School."],
         [("What school does my daughter attend?", "Your daughter goes to Washington Middle School.",
           ["Your daughter goes to Lincoln Elementary"])]),
        ("sm_upd_7",
         ["Our family doctor is Dr. Smith.", "We switched doctors. Our new doctor is Dr. Patel."],
         [("Who is our family doctor?", "Your family doctor is Dr. Patel.",
           ["Your family doctor is Dr. Smith"])]),
        ("sm_upd_8",
         ["I'm training for a 5K race.", "Changed plans. I'm now training for a full marathon."],
         [("What race am I training for?", "You are training for a full marathon.",
           ["You are training for a 5K"])]),
    ]
    for grp, setups, probes in updates:
        for idx, s in enumerate(setups):
            checks = ["belief_created"] if idx == 0 else ["tension_increased"]
            add(dom, "knowledge_update", s, ref=ref, group=grp,
                setup=True, checks=checks)
        for probe_msg, gold, forbidden in probes:
            add(dom, "knowledge_update", probe_msg, gold, ref=ref, group=grp,
                forbidden=forbidden)
    # extra standalone to reach 25
    add(dom, "knowledge_update", "My gym membership costs $40 per month.",
        ref=ref, group="sm_upd_extra", setup=True, checks=["belief_created"])


# ====================================================================
# DOMAIN 3: WORKING MEMORY (125 prompts)
# Baddeley & Hitch 1974 — multi-item, retrieval, interference, binding
# ====================================================================
def _working_memory(add):
    dom, ref = "working_memory", "Baddeley & Hitch 1974"

    # --- multi_item_encoding (25 prompts: 13 setup + 12 probes) ---
    # Store multiple items in a single turn, then recall specific ones
    items = [
        ("I need to buy eggs, bread, milk, and orange juice at the store.", "wm_multi_1",
         "What do I need from the store?", "You need to buy eggs, bread, milk, and orange juice."),
        ("My three meetings today are with Sarah at 10, Dave at 1, and the board at 3.", "wm_multi_2",
         "Who am I meeting at 1pm?", "You are meeting Dave at 1pm."),
        ("For the trip I need to pack sunscreen, hiking boots, a rain jacket, and binoculars.", "wm_multi_3",
         "What do I need to pack for the trip?", "You need to pack sunscreen, hiking boots, a rain jacket, and binoculars."),
        ("The Wi-Fi password is blueSky42!, the alarm code is 7391, and the gate code is 2580.", "wm_multi_4",
         "What is the alarm code?", "The alarm code is 7391."),
        ("My top three priorities this week are the budget report, the client demo, and hiring interviews.", "wm_multi_5",
         "What are my priorities this week?", "Your top three priorities are the budget report, the client demo, and hiring interviews."),
        ("The recipe calls for 2 cups flour, 1 cup sugar, 3 eggs, and a teaspoon of vanilla.", "wm_multi_6",
         "How much sugar does the recipe need?", "The recipe calls for 1 cup of sugar."),
        ("My kids' names are Oliver, Emma, and Lucas.", "wm_multi_7",
         "What are my kids' names?", "Your kids' names are Oliver, Emma, and Lucas."),
        ("I'm tracking three packages: one from Amazon arriving Tuesday, one from Etsy arriving Thursday, and one from eBay arriving Friday.", "wm_multi_8",
         "When does the Etsy package arrive?", "Your Etsy package arrives on Thursday."),
        ("The four cities on my road trip are Nashville, Memphis, New Orleans, and Houston.", "wm_multi_9",
         "What cities are on my road trip?", "Your road trip cities are Nashville, Memphis, New Orleans, and Houston."),
        ("My medication schedule is metformin at breakfast, lisinopril at lunch, and atorvastatin at bedtime.", "wm_multi_10",
         "When do I take atorvastatin?", "You take atorvastatin at bedtime."),
        ("The project team is me, Aisha, Carlos, Priya, and Jin.", "wm_multi_11",
         "Who is on the project team?", "The project team is you, Aisha, Carlos, Priya, and Jin."),
        ("I signed up for three classes: pottery on Monday, yoga on Wednesday, and cooking on Saturday.", "wm_multi_12",
         "When is my pottery class?", "Your pottery class is on Monday."),
    ]
    add(dom, "multi_item_encoding", "My three favorite books are Dune, Neuromancer, and Foundation.",
        ref=ref, group="wm_multi_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold in items:
        add(dom, "multi_item_encoding", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "multi_item_encoding", probe_msg, gold, ref=ref, group=grp)

    # --- item_retrieval (25 prompts: 13 setup + 12 probes) ---
    # Store set, then ask for ONE specific item
    retrieval = [
        ("My locker combination is 24-36-12.", "wm_ret_1",
         "What is the second number in my locker combination?", "The second number in your locker combination is 36."),
        ("The conference room booking is Room 405 from 2pm to 4pm.", "wm_ret_2",
         "What room is the conference in?", "The conference is in Room 405."),
        ("My flight is UA 447, gate B12, boarding at 5:15pm.", "wm_ret_3",
         "What gate is my flight at?", "Your flight is at gate B12."),
        ("The contractor quoted $15,000 for the roof, $8,000 for gutters, and $3,000 for paint.", "wm_ret_4",
         "How much did the contractor quote for gutters?", "The contractor quoted $8,000 for gutters."),
        ("My insurance policy number is AXB-229-4417.", "wm_ret_5",
         "What is my insurance policy number?", "Your insurance policy number is AXB-229-4417."),
        ("The recipe bakes at 350 degrees for 45 minutes.", "wm_ret_6",
         "What temperature should I bake at?", "You should bake at 350 degrees."),
        ("My dentist is Dr. Chen at Smile Dental, phone 555-0198.", "wm_ret_7",
         "What is my dentist's phone number?", "Your dentist's phone number is 555-0198."),
        ("The hotel reservation is at the Marriott, confirmation number MR8892, checking in March 15.", "wm_ret_8",
         "What is the hotel confirmation number?", "Your hotel confirmation number is MR8892."),
        ("The paint colors we chose are Sage Green for the bedroom, Navy for the office, and White for the kitchen.", "wm_ret_9",
         "What color did we choose for the office?", "You chose Navy for the office."),
        ("My car's VIN ends in 7J2, the plate is ABC-1234, and it's registered in Texas.", "wm_ret_10",
         "What is my license plate number?", "Your license plate is ABC-1234."),
        ("The school play is on December 12th at 7pm in the auditorium.", "wm_ret_11",
         "What time is the school play?", "The school play is at 7pm."),
        ("My new address is 742 Evergreen Terrace, Apartment 3B, Springfield IL 62704.", "wm_ret_12",
         "What apartment number do I live in?", "You live in Apartment 3B."),
    ]
    add(dom, "item_retrieval", "The server IP is 192.168.1.42, port 8080, hostname devbox.",
        ref=ref, group="wm_ret_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold in retrieval:
        add(dom, "item_retrieval", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "item_retrieval", probe_msg, gold, ref=ref, group=grp)

    # --- interference_resistance (25 prompts: 10 setup + 15 probes) ---
    # Store fact, insert distractors, then verify original survives
    interference = [
        ("wm_int_1",
         ["My password for the work VPN is Solar#99.",
          "I had pasta for lunch today. The weather was nice."],
         [("What is my work VPN password?", "Your work VPN password is Solar#99."),
          ("Did anything I mentioned change my VPN password?", "No, your VPN password is still Solar#99. The other information was unrelated."),
          ("Tell me my VPN password one more time.", "Your work VPN password is Solar#99.")]),
        ("wm_int_2",
         ["My daughter's birthday is September 3rd.",
          "I saw a great movie last night. The popcorn was stale."],
         [("When is my daughter's birthday?", "Your daughter's birthday is September 3rd."),
          ("Did the movie change anything about my daughter's birthday?", "No, the movie is unrelated. Your daughter's birthday is still September 3rd."),
          ("Remind me of my daughter's birthday.", "Your daughter's birthday is September 3rd.")]),
        ("wm_int_3",
         ["The garage door code is 4477.",
          "I need to buy a new jacket. Winter is coming."],
         [("What is the garage door code?", "The garage door code is 4477."),
          ("Has anything changed about my garage code?", "No, your garage door code is still 4477."),
          ("What's the code again?", "The garage door code is 4477.")]),
        ("wm_int_4",
         ["My next dental cleaning is February 8th.",
          "The new restaurant on Oak Street has great reviews."],
         [("When is my next dental cleaning?", "Your next dental cleaning is February 8th."),
          ("Did the restaurant info affect my appointment?", "No, the restaurant is unrelated. Your dental cleaning is still February 8th."),
          ("Confirm my dental appointment date.", "Your dental cleaning is scheduled for February 8th.")]),
        ("wm_int_5",
         ["I owe my brother $200 from last weekend.",
          "Traffic was terrible today. I hate the commute."],
         [("How much do I owe my brother?", "You owe your brother $200 from last weekend."),
          ("Did my commute change what I owe?", "No, your commute is unrelated. You still owe your brother $200."),
          ("What's the exact amount I owe?", "You owe your brother $200.")]),
    ]
    for grp, setups, probes in interference:
        for s in setups:
            add(dom, "interference_resistance", s, ref=ref, group=grp,
                setup=True, checks=["belief_created"])
        for probe_msg, gold in probes:
            add(dom, "interference_resistance", probe_msg, gold, ref=ref, group=grp)

    # --- feature_binding (25 prompts: 13 setup + 12 probes) ---
    # Store item+attribute pairs, then ask for the attribute of a specific item
    binding = [
        ("Sarah drives a red Tesla, Mike drives a black BMW, and Lisa drives a white Prius.", "wm_bind_1",
         "What color is Mike's car?", "Mike drives a black BMW."),
        ("The upstairs bathroom has blue tiles, the downstairs has white, and the master has gray.", "wm_bind_2",
         "What color tiles are in the master bathroom?", "The master bathroom has gray tiles."),
        ("Monday's dinner is tacos, Tuesday is stir fry, Wednesday is soup.", "wm_bind_3",
         "What's for dinner on Tuesday?", "Tuesday's dinner is stir fry."),
        ("Alice sits in cubicle 201, Bob in 305, and Carol in 112.", "wm_bind_4",
         "Which cubicle is Bob in?", "Bob sits in cubicle 305."),
        ("The red folder has tax documents, the blue folder has medical records, the green folder has insurance.", "wm_bind_5",
         "What's in the blue folder?", "The blue folder has medical records."),
        ("Plant A needs watering daily, Plant B every 3 days, Plant C weekly.", "wm_bind_6",
         "How often does Plant B need water?", "Plant B needs watering every 3 days."),
        ("Jake's birthday is in March, Lily's in July, and Noah's in November.", "wm_bind_7",
         "When is Lily's birthday?", "Lily's birthday is in July."),
        ("The small box weighs 5 lbs, the medium 15 lbs, and the large 40 lbs.", "wm_bind_8",
         "How much does the medium box weigh?", "The medium box weighs 15 lbs."),
        ("Morning shift starts at 6am, afternoon at 2pm, and night at 10pm.", "wm_bind_9",
         "What time does the afternoon shift start?", "The afternoon shift starts at 2pm."),
        ("Route A takes 25 minutes, Route B takes 40 minutes, Route C takes 15 minutes but has tolls.", "wm_bind_10",
         "What's the trade-off with Route C?", "Route C is the fastest at 15 minutes but it has tolls."),
        ("In my team, Emily handles frontend, Marcus handles backend, and Raj handles DevOps.", "wm_bind_11",
         "Who handles backend?", "Marcus handles backend."),
        ("My morning coffee is a latte, afternoon is green tea, evening is chamomile.", "wm_bind_12",
         "What do I drink in the evening?", "You drink chamomile in the evening."),
    ]
    add(dom, "feature_binding", "Office A has a projector, Office B has a whiteboard, Office C has a video setup.",
        ref=ref, group="wm_bind_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold in binding:
        add(dom, "feature_binding", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "feature_binding", probe_msg, gold, ref=ref, group=grp)

    # --- updating (25 prompts: 10 setup + 15 probes) ---
    # Store set, update one item, verify update + verify others unchanged
    updating = [
        ("wm_upd_1",
         ["My three errands today are: dry cleaning, post office, and hardware store.",
          "Actually, skip the post office. Replace it with the pharmacy."],
         [("What are my errands today?", "Your errands are dry cleaning, pharmacy, and hardware store."),
          ("Am I still going to the hardware store?", "Yes, the hardware store is still on your errand list."),
          ("Am I going to the post office?", "No, you replaced the post office with the pharmacy.")]),
        ("wm_upd_2",
         ["The meeting attendees are John, Maria, and Feng.",
          "Maria can't make it. Replace her with Derek."],
         [("Who is attending the meeting?", "The meeting attendees are John, Derek, and Feng."),
          ("Is John still attending?", "Yes, John is still attending the meeting."),
          ("Is Maria attending?", "No, Maria can't make it and was replaced by Derek.")]),
        ("wm_upd_3",
         ["Tonight's dinner plan: appetizer is bruschetta, main is salmon, dessert is tiramisu.",
          "Change the main course from salmon to lamb chops."],
         [("What's the main course tonight?", "The main course is lamb chops."),
          ("What about the appetizer?", "The appetizer is still bruschetta."),
          ("What's for dessert?", "Dessert is tiramisu.")]),
        ("wm_upd_4",
         ["Scores: Team A has 12 points, Team B has 8 points, Team C has 15 points.",
          "Update: Team B scored 5 more points."],
         [("How many points does Team B have now?", "Team B now has 13 points."),
          ("What about Team A?", "Team A still has 12 points."),
          ("Who is in the lead?", "Team C is in the lead with 15 points.")]),
        ("wm_upd_5",
         ["My weekly schedule: Monday is gym, Wednesday is book club, Friday is date night.",
          "I'm moving gym from Monday to Tuesday."],
         [("When am I going to the gym this week?", "You are going to the gym on Tuesday."),
          ("What am I doing Wednesday?", "Wednesday is book club."),
          ("Is date night still Friday?", "Yes, date night is still on Friday.")]),
    ]
    for grp, setups, probes in updating:
        for s in setups:
            add(dom, "updating", s, ref=ref, group=grp,
                setup=True, checks=["belief_created"])
        for probe_msg, gold in probes:
            add(dom, "updating", probe_msg, gold, ref=ref, group=grp)

# ====================================================================
# DOMAIN 4: SELECTIVE ATTENTION (125 prompts)
# Broadbent 1958, Posner 1980 — filtering, gating, inhibition
# ====================================================================
def _selective_attention(add):
    dom, ref = "selective_attention", "Broadbent 1958"

    # --- target_encoding (25 prompts: 13 setup + 12 probes) ---
    # Embed target fact in noisy context, then recall just the target
    targets = [
        ("I ramble a lot but the important thing is my flight is at 7pm. Anyway the food was great.", "sa_tgt_1",
         "What time is my flight?", "Your flight is at 7pm."),
        ("So I was chatting about movies and oh right my doctor said my cholesterol is 220. Random tangent about cats.", "sa_tgt_2",
         "What is my cholesterol level?", "Your doctor said your cholesterol is 220."),
        ("Weather's weird this year. Anyway the lease renewal rate is $1,800 per month. Did you see the game?", "sa_tgt_3",
         "What's my lease renewal rate?", "Your lease renewal rate is $1,800 per month."),
        ("Had coffee this morning, talked about politics, and oh — my son's school starts at 8:15am now.", "sa_tgt_4",
         "What time does my son's school start?", "Your son's school starts at 8:15am."),
        ("Lots going on. The key thing: the server migration is scheduled for March 20th. Also I like tacos.", "sa_tgt_5",
         "When is the server migration?", "The server migration is scheduled for March 20th."),
        ("I keep forgetting things but the critical one is my prescription refill is due on the 5th.", "sa_tgt_6",
         "When is my prescription refill due?", "Your prescription refill is due on the 5th."),
        ("Thinking out loud here... the mortgage rate they offered is 6.2%. Where was I going with this.", "sa_tgt_7",
         "What mortgage rate was I offered?", "You were offered a mortgage rate of 6.2%."),
        ("Between the traffic and the noise, the point is my interview is at 2:30pm on Thursday.", "sa_tgt_8",
         "When is my interview?", "Your interview is at 2:30pm on Thursday."),
        ("Grocery list, random thoughts... oh and the daycare tuition is going up to $1,200 next month.", "sa_tgt_9",
         "How much will daycare cost next month?", "Daycare tuition is going up to $1,200 next month."),
        ("I was thinking about the weekend and realized my passport expires June 2027.", "sa_tgt_10",
         "When does my passport expire?", "Your passport expires June 2027."),
        ("So much happening. Key info: the contractor's bid for the bathroom is $12,500.", "sa_tgt_11",
         "How much did the contractor bid?", "The contractor bid $12,500 for the bathroom."),
        ("Anyway amid all this chaos, my daughter's recital is May 3rd at 6pm.", "sa_tgt_12",
         "When is my daughter's recital?", "Your daughter's recital is May 3rd at 6pm."),
    ]
    add(dom, "target_encoding", "Bunch of noise but my car registration expires April 30th. Also thinking about lunch.",
        ref=ref, group="sa_tgt_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold in targets:
        add(dom, "target_encoding", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "target_encoding", probe_msg, gold, ref=ref, group=grp)

    # --- distractor_filtering (25 prompts: 10 setup + 15 probes) ---
    # Store multiple facts across sessions, ask targeted question
    distractors = [
        ("sa_dist_1",
         ["I love Italian food, my favorite is margherita pizza.",
          "My work project deadline is November 15th."],
         [("When is my project deadline?", "Your project deadline is November 15th.",
           ["pizza", "Italian food"]),
          ("What's my project deadline again? Just the date.", "Your project deadline is November 15th.",
           ["pizza", "margherita"]),
          ("Remind me about the deadline.", "Your project deadline is November 15th.",
           ["food", "Italian"])]),
        ("sa_dist_2",
         ["My cat's name is Mittens and she's very playful.",
          "The quarterly review is on the 22nd at 3pm."],
         [("When is the quarterly review?", "The quarterly review is on the 22nd at 3pm.",
           ["Mittens", "cat"]),
          ("What time is the review?", "The quarterly review is at 3pm on the 22nd.",
           ["cat", "playful"]),
          ("Tell me about the quarterly review.", "Your quarterly review is on the 22nd at 3pm.",
           ["Mittens", "pet"])]),
        ("sa_dist_3",
         ["I went skydiving last summer, it was amazing.",
          "My car insurance premium is $180 per month."],
         [("How much is my car insurance?", "Your car insurance premium is $180 per month.",
           ["skydiving", "amazing"]),
          ("What do I pay monthly for insurance?", "You pay $180 per month for car insurance.",
           ["skydiving", "summer"]),
          ("Exact cost of my auto insurance?", "Your car insurance is $180 per month.",
           ["jump", "sky"])]),
    ]
    for grp, setups, probes in distractors:
        for s in setups:
            add(dom, "distractor_filtering", s, ref=ref, group=grp,
                setup=True, checks=["belief_created"])
        for probe_msg, gold, forbidden in probes:
            add(dom, "distractor_filtering", probe_msg, gold, ref=ref,
                group=grp, forbidden=forbidden)
    # additional standalone pairs to reach 25
    extras_df = [
        ("I collect vintage stamps. My rent is $1,500 per month.", "sa_dist_4",
         "How much is my rent?", "Your rent is $1,500 per month.", ["stamps", "vintage"]),
        ("My hobby is woodworking. The team standup is at 9am.", "sa_dist_5",
         "When is the team standup?", "The team standup is at 9am.", ["woodworking"]),
        ("I ran a 5K last month. My kid's tutor charges $50 per hour.", "sa_dist_6",
         "How much does the tutor charge?", "Your kid's tutor charges $50 per hour.", ["running", "5K"]),
        ("I'm into birdwatching. My flight home is on United at 4pm.", "sa_dist_7",
         "What time is my flight home?", "Your flight home is at 4pm on United.", ["birdwatching", "birds"]),
        ("I love gardening. The dentist copay is $35.", "sa_dist_8",
         "What is the dentist copay?", "The dentist copay is $35.", ["gardening", "garden"]),
    ]
    for setup_msg, grp, probe_msg, gold, forbidden in extras_df:
        add(dom, "distractor_filtering", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "distractor_filtering", probe_msg, gold, ref=ref,
            group=grp, forbidden=forbidden)

    # --- inhibition (25 prompts: 10 setup + 15 probes) ---
    # Prime with topic Y, then ask about topic X — verify no cross-bleed
    inhibition = [
        ("sa_inh_1",
         ["My brother is a chef who specializes in French cuisine.",
          "My sister is a software developer at Google."],
         [("What does my sister do?", "Your sister is a software developer at Google.",
           ["chef", "French cuisine", "cooking"]),
          ("Where does my sister work?", "Your sister works at Google.",
           ["restaurant", "kitchen"]),
          ("Is my sister a chef?", "No, your sister is a software developer at Google. It's your brother who is a chef.",
           [])]),
        ("sa_inh_2",
         ["I play tennis every Saturday morning.",
          "I have piano lessons on Wednesday evenings."],
         [("What do I do on Wednesday evenings?", "You have piano lessons on Wednesday evenings.",
           ["tennis", "Saturday"]),
          ("When do I play tennis?", "You play tennis every Saturday morning.",
           ["piano", "Wednesday"]),
          ("Do I play tennis on Wednesday?", "No, Wednesday is for piano lessons. You play tennis on Saturday mornings.",
           [])]),
        ("sa_inh_3",
         ["My checking account is at Chase with $3,200 balance.",
          "My savings account is at Ally with $15,000 balance."],
         [("How much is in my savings?", "Your savings account at Ally has $15,000.",
           ["Chase", "$3,200", "checking"]),
          ("Where is my checking account?", "Your checking account is at Chase.",
           ["Ally", "savings", "$15,000"]),
          ("Is my savings at Chase?", "No, your savings is at Ally. Your checking account is at Chase.",
           [])]),
        ("sa_inh_4",
         ["For breakfast I always have oatmeal with berries.",
          "For lunch I usually have a turkey sandwich."],
         [("What do I eat for lunch?", "You usually have a turkey sandwich for lunch.",
           ["oatmeal", "berries", "breakfast"]),
          ("What's my breakfast routine?", "You always have oatmeal with berries for breakfast.",
           ["turkey", "sandwich", "lunch"]),
          ("Do I eat oatmeal for lunch?", "No, you eat oatmeal for breakfast. For lunch you have a turkey sandwich.",
           [])]),
        ("sa_inh_5",
         ["My morning alarm is set for 6am.",
          "My evening alarm for medication is at 9pm."],
         [("What time is my medication alarm?", "Your medication alarm is at 9pm.",
           ["6am", "morning"]),
          ("When is my morning alarm?", "Your morning alarm is set for 6am.",
           ["9pm", "medication"]),
          ("Is my morning alarm at 9pm?", "No, your morning alarm is at 6am. The 9pm alarm is for medication.",
           [])]),
    ]
    for grp, setups, probes in inhibition:
        for s in setups:
            add(dom, "inhibition", s, ref=ref, group=grp,
                setup=True, checks=["belief_created"])
        for probe_msg, gold, forbidden in probes:
            add(dom, "inhibition", probe_msg, gold, ref=ref, group=grp,
                forbidden=forbidden)

    # --- relevance_gating (25 prompts: 13 setup + 12 probes) ---
    # General knowledge questions that should NOT pull in personal beliefs
    gate = [
        ("I think Mars is the most fascinating planet.", "sa_gate_1",
         "What is the closest planet to the sun?", "Mercury is the closest planet to the sun.",
         ["Mars", "fascinating"]),
        ("I believe exercise is the best medicine.", "sa_gate_2",
         "What organ pumps blood through the body?", "The heart pumps blood through the body.",
         ["exercise", "best medicine"]),
        ("My favorite number is 7.", "sa_gate_3",
         "What is the square root of 144?", "The square root of 144 is 12.",
         ["favorite number", "7"]),
        ("I think Python is the best language ever.", "sa_gate_4",
         "Who invented the C programming language?", "Dennis Ritchie invented the C programming language.",
         ["Python", "best language"]),
        ("I love watching documentaries about World War 2.", "sa_gate_5",
         "In what year did World War 1 begin?", "World War 1 began in 1914.",
         ["World War 2", "documentaries"]),
        ("I'm obsessed with Japanese culture.", "sa_gate_6",
         "What is the capital of France?", "The capital of France is Paris.",
         ["Japan", "Japanese"]),
        ("I believe artificial intelligence will change everything.", "sa_gate_7",
         "Who wrote the novel 1984?", "George Orwell wrote the novel 1984.",
         ["artificial intelligence", "AI"]),
        ("My favorite sport is basketball.", "sa_gate_8",
         "How many players are on a soccer team?", "A soccer team has 11 players on the field.",
         ["basketball"]),
        ("I think coffee is better than tea.", "sa_gate_9",
         "What country produces the most tea?", "China is the world's largest tea producer.",
         ["coffee", "better than tea"]),
        ("I've always been interested in quantum physics.", "sa_gate_10",
         "What is Newton's first law of motion?", "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion unless acted on by an external force.",
         ["quantum", "interested"]),
        ("I adore Italian food.", "sa_gate_11",
         "What country is sushi from?", "Sushi originates from Japan.",
         ["Italian", "adore"]),
        ("I think democracy is the best form of government.", "sa_gate_12",
         "Who was the first emperor of Rome?", "Augustus was the first emperor of Rome.",
         ["democracy"]),
    ]
    add(dom, "relevance_gating", "I believe space exploration is humanity's greatest endeavor.",
        ref=ref, group="sa_gate_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, forbidden in gate:
        add(dom, "relevance_gating", setup_msg, ref=ref, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "relevance_gating", probe_msg, gold, ref=ref, group=grp,
            forbidden=forbidden)

    # --- focused_retrieval (25 prompts: 10 setup + 15 probes) ---
    # Store complex multipart scenarios, retrieve one specific detail
    focused = [
        ("sa_foc_1",
         ["I'm planning a trip to Barcelona from March 10-17. We're staying at Hotel Arts, room 412. The flight is on Delta, DL445, departing at noon.",
          "We have reservations at La Sagrada Familia on the 12th and Park Guell on the 14th."],
         [("What room are we staying in?", "You are staying in room 412 at Hotel Arts in Barcelona."),
          ("What is our flight number?", "Your flight number is Delta DL445."),
          ("When is the Park Guell reservation?", "Your Park Guell reservation is on the 14th.")]),
        ("sa_foc_2",
         ["The home inspection found: roof needs repair ($5K), furnace is old but working, plumbing is fine, electrical panel needs upgrading ($3K).",
          "The seller agreed to cover the roof but not the electrical."],
         [("What did the seller agree to cover?", "The seller agreed to cover the roof repair."),
          ("How much is the electrical panel upgrade?", "The electrical panel upgrade costs $3,000."),
          ("Is the plumbing okay?", "Yes, the inspection found the plumbing is fine.")]),
        ("sa_foc_3",
         ["My investment portfolio: 40% index funds, 30% bonds, 20% individual stocks, 10% crypto.",
          "The index funds returned 12% last year, bonds 4%, stocks lost 3%, crypto gained 25%."],
         [("What percentage of my portfolio is in bonds?", "30% of your portfolio is in bonds."),
          ("How did my individual stocks perform?", "Your individual stocks lost 3% last year."),
          ("What returned the most?", "Crypto gained the most at 25% last year.")]),
        ("sa_foc_4",
         ["The party plans: Saturday at 6pm, venue is Lakeview Hall, caterer is Fresh Bites, DJ is SpinMaster Dave, budget is $4,000.",
          "Guest list is 50 people. Theme is tropical."],
         [("Who is the DJ?", "The DJ is SpinMaster Dave."),
          ("What's the party budget?", "The party budget is $4,000."),
          ("How many guests?", "The guest list is 50 people.")]),
        ("sa_foc_5",
         ["Car maintenance needed: oil change ($50), new tires ($600), brake pads ($300), alignment ($100).",
          "The mechanic can fit me in on Tuesday morning. Total estimate is $1,050."],
         [("How much are the new tires?", "New tires cost $600."),
          ("When can the mechanic see me?", "The mechanic can fit you in on Tuesday morning."),
          ("What's the total maintenance estimate?", "The total estimate is $1,050.")]),
    ]
    for grp, setups, probes in focused:
        for s in setups:
            add(dom, "focused_retrieval", s, ref=ref, group=grp,
                setup=True, checks=["belief_created"])
        for probe_msg, gold in probes:
            add(dom, "focused_retrieval", probe_msg, gold, ref=ref, group=grp)

# ====================================================================
# DOMAIN 5: LANGUAGE COMPREHENSION (125 prompts)
# Grice 1975, Searle 1975 — implicature, speech acts, presupposition
# ====================================================================
def _language_comprehension(add):
    dom = "language_comprehension"

    # --- scalar_implicature (25 prompts) ---
    # Grice: "some" implies "not all", "good" implies "not great"
    scalars = [
        ("Some of the students passed the exam.", "lc_scal_1",
         "Did all of the students pass?",
         "Saying 'some' students passed implies that not all of them passed. Some likely failed.",
         "Grice 1975", ["All of the students passed"]),
        ("The movie was good.", "lc_scal_2",
         "Would you say the movie was great?",
         "Describing it as 'good' suggests it was decent but not exceptional or great.",
         "Grice 1975", ["The movie was great", "It was an excellent film"]),
        ("I finished some of the report.", "lc_scal_3",
         "Did I finish the entire report?",
         "No, saying you finished 'some' of the report implies you did not finish all of it.",
         "Grice 1975", ["You finished the entire report"]),
        ("A few people showed up to the meeting.", "lc_scal_4",
         "Was the meeting well attended?",
         "No, 'a few' implies a small number of people attended, so the meeting was not well attended.",
         "Grice 1975", ["The meeting was well attended", "Many people came"]),
        ("The restaurant was okay.", "lc_scal_5",
         "Would you recommend the restaurant?",
         "Describing it as just 'okay' suggests it was mediocre, not particularly worth recommending.",
         "Grice 1975", ["It was a wonderful restaurant", "Highly recommended"]),
        ("I sometimes go to the gym.", "lc_scal_6",
         "Am I a regular gym-goer?",
         "Saying 'sometimes' implies you do not go regularly or frequently to the gym.",
         "Grice 1975", ["You go to the gym regularly"]),
        ("Most of the team agreed with the proposal.", "lc_scal_7",
         "Did the entire team agree?",
         "No, 'most' implies that while the majority agreed, not everyone did. Some team members likely disagreed.",
         "Grice 1975", ["The entire team agreed"]),
        ("The presentation was adequate.", "lc_scal_8",
         "Was the presentation impressive?",
         "Calling it 'adequate' suggests it met minimum expectations but was not impressive or outstanding.",
         "Grice 1975", ["The presentation was impressive"]),
        ("He is a competent programmer.", "lc_scal_9",
         "Is he an exceptional programmer?",
         "Describing him as 'competent' implies he is capable but not exceptional or outstanding.",
         "Grice 1975", ["He is an exceptional programmer"]),
        ("I ate a bit of the cake.", "lc_scal_10",
         "Did I eat a lot of cake?",
         "No, 'a bit' implies you ate only a small amount of the cake.",
         "Grice 1975", ["You ate a lot of cake"]),
        ("The weather is acceptable for the picnic.", "lc_scal_11",
         "Is the weather perfect for the picnic?",
         "Describing the weather as 'acceptable' implies it's not ideal or perfect, just tolerable enough.",
         "Grice 1975", ["The weather is perfect"]),
        ("I understood parts of the lecture.", "lc_scal_12",
         "Did I understand the whole lecture?",
         "No, 'parts of' implies you did not understand the entire lecture, only some portions.",
         "Grice 1975", ["You understood everything"]),
    ]
    # one standalone setup
    add(dom, "scalar_implicature", "I managed to solve a couple of the problems on the test.",
        ref="Grice 1975", group="lc_scal_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, reference, forbidden in scalars:
        add(dom, "scalar_implicature", setup_msg, ref=reference, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "scalar_implicature", probe_msg, gold, ref=reference, group=grp,
            forbidden=forbidden)

    # --- indirect_speech (25 prompts) ---
    # Searle: utterances that mean something different from literal content
    indirects = [
        ("I told my assistant 'It's cold in here' while looking at the window.", "lc_ind_1",
         "What was I really asking my assistant to do?",
         "You were indirectly asking your assistant to close the window or adjust the temperature, not just stating a fact about the temperature.",
         "Searle 1975", ["You were simply observing the temperature"]),
        ("My wife said 'The trash is full' while I was on the couch.", "lc_ind_2",
         "What did my wife actually want?",
         "Your wife was indirectly requesting that you take out the trash, not merely making an observation.",
         "Searle 1975", ["She was just commenting on the trash"]),
        ("During the meeting my boss said 'Does anyone know what time it is?'", "lc_ind_3",
         "What was my boss implying?",
         "Your boss was implying the meeting was running too long and it was time to wrap up, not genuinely asking for the time.",
         "Searle 1975", ["Your boss wanted to know the current time"]),
        ("The hostess said 'We have a lovely patio available' when the indoor section was packed.", "lc_ind_4",
         "What was the hostess really saying?",
         "The hostess was suggesting you sit on the patio since there are no indoor tables available.",
         "Searle 1975", ["She was genuinely excited about the patio"]),
        ("My coworker said 'I wish someone would refill the coffee pot' while looking at me.", "lc_ind_5",
         "What did my coworker want?",
         "Your coworker was indirectly asking you to refill the coffee pot.",
         "Searle 1975", ["They were just expressing a general wish"]),
        ("The teacher told the class 'I notice the homework box is quite empty today.'", "lc_ind_6",
         "What was the teacher communicating?",
         "The teacher was pointing out that many students did not turn in their homework, implying disapproval or a reminder to submit it.",
         "Searle 1975", ["The teacher was happy about the empty box"]),
        ("My friend said 'I hear that restaurant has amazing desserts' as we were deciding where to eat.", "lc_ind_7",
         "What was my friend really suggesting?",
         "Your friend was indirectly suggesting you should eat at that restaurant.",
         "Searle 1975", ["They were randomly sharing a food review"]),
        ("My mom said 'Your room is looking very... creative' with a raised eyebrow.", "lc_ind_8",
         "What did my mom mean?",
         "Your mom was politely saying your room is messy and needs to be cleaned up. 'Creative' was a euphemism for disorganized.",
         "Searle 1975", ["She was genuinely complimenting your room arrangement"]),
        ("The dentist said 'Have you been flossing?' after looking at my gums.", "lc_ind_9",
         "What was the dentist's real message?",
         "The dentist was implying your gums show signs that you haven't been flossing regularly, not genuinely asking.",
         "Searle 1975", ["The dentist was curious about your habits"]),
        ("My neighbor said 'Your dog was in my yard again this morning.'", "lc_ind_10",
         "What was my neighbor asking me to do?",
         "Your neighbor was asking you to keep your dog out of their yard. It was a complaint disguised as a statement.",
         "Searle 1975", ["They were simply informing you of an observation"]),
        ("During the budget review, the CFO said 'These numbers are... ambitious.'", "lc_ind_11",
         "What did the CFO actually think of the numbers?",
         "The CFO was implying the budget projections are unrealistic or too optimistic, not genuinely praising your ambition.",
         "Searle 1975", ["The CFO was impressed by your ambition"]),
        ("My partner said 'I see the dishes from breakfast are still in the sink.'", "lc_ind_12",
         "What was my partner really saying?",
         "Your partner was asking you to wash the dishes, not just making an observation about the kitchen.",
         "Searle 1975", ["They were merely commenting on the dishes"]),
    ]
    add(dom, "indirect_speech", "My professor said 'That's one way to solve it' after I presented my solution.",
        ref="Searle 1975", group="lc_ind_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, reference, forbidden in indirects:
        add(dom, "indirect_speech", setup_msg, ref=reference, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "indirect_speech", probe_msg, gold, ref=reference, group=grp,
            forbidden=forbidden)

    # --- presupposition (25 prompts: 13 setup + 12 probes) ---
    presup = [
        ("I stopped running marathons last year.", "lc_pres_1",
         "Did I use to run marathons?",
         "Yes, saying you 'stopped' running marathons presupposes that you used to run them.",
         "Stalnaker 1974"),
        ("I regret selling my vintage guitar.", "lc_pres_2",
         "Did I own a vintage guitar?",
         "Yes, saying you 'regret selling' it presupposes that you owned a vintage guitar and sold it.",
         "Stalnaker 1974"),
        ("I quit smoking five years ago.", "lc_pres_3",
         "Was I a smoker?",
         "Yes, saying you 'quit' smoking presupposes that you were previously a smoker.",
         "Stalnaker 1974"),
        ("I miss living in San Francisco.", "lc_pres_4",
         "Did I live in San Francisco?",
         "Yes, saying you 'miss living' there presupposes that you previously lived in San Francisco.",
         "Stalnaker 1974"),
        ("I returned the defective laptop to the store.", "lc_pres_5",
         "Was there something wrong with the laptop?",
         "Yes, describing the laptop as 'defective' presupposes it had a fault or problem.",
         "Stalnaker 1974"),
        ("My ex-wife and I still co-parent well.", "lc_pres_6",
         "Was I previously married?",
         "Yes, referring to your 'ex-wife' presupposes you were previously married.",
         "Stalnaker 1974"),
        ("I finally finished writing my novel.", "lc_pres_7",
         "Was I working on a novel?",
         "Yes, saying you 'finally finished' presupposes you had been working on writing a novel.",
         "Stalnaker 1974"),
        ("I switched back to Android after trying iPhone.", "lc_pres_8",
         "Did I use Android before iPhone?",
         "Yes, 'switched back to Android' presupposes you used Android before, then tried iPhone, and returned.",
         "Stalnaker 1974"),
        ("I overcame my fear of public speaking.", "lc_pres_9",
         "Was I afraid of public speaking?",
         "Yes, 'overcame my fear' presupposes that you previously had a fear of public speaking.",
         "Stalnaker 1974"),
        ("I got my driver's license reinstated.", "lc_pres_10",
         "Was my license suspended at some point?",
         "Yes, 'reinstated' presupposes your driver's license had been suspended or revoked.",
         "Stalnaker 1974"),
        ("I recovered from the surgery faster than expected.", "lc_pres_11",
         "Did I have surgery?",
         "Yes, 'recovered from the surgery' presupposes that you underwent a surgical procedure.",
         "Stalnaker 1974"),
        ("I forgave my brother for what he said at Thanksgiving.", "lc_pres_12",
         "Did my brother say something hurtful?",
         "Yes, 'forgave' presupposes your brother said something offensive or hurtful at Thanksgiving.",
         "Stalnaker 1974"),
    ]
    add(dom, "presupposition", "I managed to pass the bar exam on my second attempt.",
        ref="Stalnaker 1974", group="lc_pres_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, reference in presup:
        add(dom, "presupposition", setup_msg, ref=reference, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "presupposition", probe_msg, gold, ref=reference, group=grp)

    # --- figurative_language (25 prompts: 13 setup + 12 probes) ---
    figurative = [
        ("My coworker said the new policy is a ticking time bomb.", "lc_fig_1",
         "What does my coworker think about the policy?",
         "Your coworker believes the new policy is dangerous and could cause serious problems or blow up eventually.",
         "Lakoff & Johnson 1980", ["Your coworker thinks there is a literal bomb"]),
        ("My coach said I need to hit the ground running this season.", "lc_fig_2",
         "What does my coach want me to do?",
         "Your coach wants you to start the season strong and make progress immediately from the beginning.",
         "Lakoff & Johnson 1980", ["Your coach wants you to literally run"]),
        ("My manager said we need to think outside the box on this project.", "lc_fig_3",
         "What is my manager asking for?",
         "Your manager is asking for creative, unconventional, or innovative approaches to the project.",
         "Lakoff & Johnson 1980", ["There is an actual box"]),
        ("My friend said her new job is a breath of fresh air.", "lc_fig_4",
         "How does my friend feel about her new job?",
         "Your friend feels her new job is refreshing and a welcome change from her previous situation.",
         "Lakoff & Johnson 1980", ["She is literally breathing fresh air"]),
        ("The professor said the exam was a piece of cake.", "lc_fig_5",
         "How hard was the exam?",
         "The professor is saying the exam was very easy.",
         "Lakoff & Johnson 1980", ["The exam involved actual cake"]),
        ("My therapist said I'm carrying the weight of the world on my shoulders.", "lc_fig_6",
         "What is my therapist saying about my stress?",
         "Your therapist is saying you are burdened with too much stress, responsibility, or worry.",
         "Lakoff & Johnson 1980", ["You are literally carrying something heavy"]),
        ("My dad said the stock market is on a roller coaster lately.", "lc_fig_7",
         "What is my dad saying about the market?",
         "Your dad is saying the stock market has been very volatile with dramatic ups and downs.",
         "Lakoff & Johnson 1980", ["The stock market is literally a ride"]),
        ("My colleague said she's burning the candle at both ends.", "lc_fig_8",
         "What is my colleague saying about herself?",
         "Your colleague is saying she is overworking herself and exhausting her energy from both sides, likely working too much and sleeping too little.",
         "Lakoff & Johnson 1980", ["She is literally burning candles"]),
        ("My neighbor said their renovation is a money pit.", "lc_fig_9",
         "What does my neighbor mean?",
         "Your neighbor means the renovation is continuously consuming large amounts of money with no end in sight.",
         "Lakoff & Johnson 1980", ["There is a literal pit of money"]),
        ("The client said our proposal is music to their ears.", "lc_fig_10",
         "How does the client feel about our proposal?",
         "The client is very pleased and delighted by your proposal.",
         "Lakoff & Johnson 1980", ["The client heard actual music"]),
        ("My sister said her kids are a handful.", "lc_fig_11",
         "What is my sister saying about her children?",
         "Your sister is saying her kids are difficult to manage and require a lot of energy and attention.",
         "Lakoff & Johnson 1980", ["She is literally holding her children"]),
        ("The CEO said the company is at a crossroads.", "lc_fig_12",
         "What does the CEO mean about the company's situation?",
         "The CEO means the company is at a critical decision point where it must choose between different strategic directions.",
         "Lakoff & Johnson 1980", ["The company is at a literal road intersection"]),
    ]
    add(dom, "figurative_language", "My mechanic said the engine is on its last legs.",
        ref="Lakoff & Johnson 1980", group="lc_fig_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, reference, forbidden in figurative:
        add(dom, "figurative_language", setup_msg, ref=reference, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "figurative_language", probe_msg, gold, ref=reference, group=grp,
            forbidden=forbidden)

    # --- disambiguation (25 prompts: 13 setup + 12 probes) ---
    disambig = [
        ("I went to the bank to deposit a check.", "lc_dis_1",
         "What kind of bank did I go to?",
         "You went to a financial bank to deposit a check, not a river bank.",
         "Grice 1975", ["river bank"]),
        ("I need to get a new bat for the game on Saturday.", "lc_dis_2",
         "What kind of bat do I need?",
         "You need a baseball or cricket bat for the sports game, not the animal.",
         "Grice 1975", ["flying mammal"]),
        ("The pitcher threw a perfect game last night.", "lc_dis_3",
         "What kind of pitcher?",
         "A baseball pitcher threw a perfect game, not a container for liquid.",
         "Grice 1975", ["jug", "container"]),
        ("I need to change the bulb in the living room lamp.", "lc_dis_4",
         "What kind of bulb?",
         "You need to change a light bulb in your lamp, not a plant bulb like a tulip bulb.",
         "Grice 1975", ["plant bulb", "tulip"]),
        ("She has a very sharp tongue in meetings.", "lc_dis_5",
         "What does 'sharp tongue' mean here?",
         "It means she speaks in a direct, cutting, or critical way during meetings. Not literally sharp.",
         "Grice 1975", ["Her tongue is literally sharp"]),
        ("I left my mouse at the office.", "lc_dis_6",
         "What kind of mouse?",
         "You left your computer mouse at the office, not a live animal.",
         "Grice 1975", ["rodent", "animal mouse"]),
        ("The crane lifted the steel beam into place.", "lc_dis_7",
         "What kind of crane?",
         "A construction crane lifted the steel beam, not the bird.",
         "Grice 1975", ["bird"]),
        ("I booked a suite at the hotel for our anniversary.", "lc_dis_8",
         "What kind of suite?",
         "You booked a hotel suite, which is a set of connected rooms, for your anniversary.",
         "Grice 1975", ["sweet", "candy"]),
        ("The seal on the document looks official.", "lc_dis_9",
         "What kind of seal?",
         "An official stamp or seal on the document, not the marine animal.",
         "Grice 1975", ["animal"]),
        ("I need to iron my suit before the presentation.", "lc_dis_10",
         "What am I doing?",
         "You need to press wrinkles out of your business suit using a clothing iron before your presentation.",
         "Grice 1975", ["iron metal", "lawsuit"]),
        ("The current swept the kayak downstream.", "lc_dis_11",
         "What kind of current?",
         "A water current in the river swept the kayak downstream, not electrical current.",
         "Grice 1975", ["electrical"]),
        ("I saw a bass jump out of the lake.", "lc_dis_12",
         "What kind of bass?",
         "A bass fish jumped out of the lake, not a bass guitar or bass voice.",
         "Grice 1975", ["guitar", "musical instrument"]),
    ]
    add(dom, "disambiguation", "The lead actor delivered a powerful performance.",
        ref="Grice 1975", group="lc_dis_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, reference, forbidden in disambig:
        add(dom, "disambiguation", setup_msg, ref=reference, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "disambiguation", probe_msg, gold, ref=reference, group=grp,
            forbidden=forbidden)

# ====================================================================
# DOMAIN 6: REASONING (125 prompts)
# Wason 1966, Gentner 1983, Tversky & Kahneman 1974
# ====================================================================
def _reasoning(add):
    dom = "reasoning"

    # --- modus_ponens (25 prompts: 13 setup + 12 probes) ---
    # If P then Q. P is true. → Q is true.
    mp = [
        ("If it rains, the soccer game will be canceled. It is raining.", "r_mp_1",
         "Will the soccer game be played?",
         "No, the soccer game will be canceled because it is raining.",
         "Wason 1966", ["The game will be played"]),
        ("If the temperature drops below 32°F, the pipes will freeze. The forecast says 25°F tonight.", "r_mp_2",
         "Should I worry about the pipes?",
         "Yes, the pipes will freeze because 25°F is below 32°F.",
         "Wason 1966", ["The pipes will be fine"]),
        ("If a student scores above 90, they get an A. Maria scored 95.", "r_mp_3",
         "What grade does Maria get?",
         "Maria gets an A because she scored 95, which is above 90.",
         "Wason 1966", ["Maria does not get an A"]),
        ("If the store is closed, I'll order delivery. The store closed at 8pm and it's 9pm now.", "r_mp_4",
         "What should I do for food?",
         "You should order delivery because the store is closed.",
         "Wason 1966", ["Go to the store"]),
        ("If you have a fever over 101°F, you should stay home. My temperature is 103°F.", "r_mp_5",
         "Should I go to work?",
         "No, you should stay home because your temperature is 103°F, which is over 101°F.",
         "Wason 1966", ["You should go to work"]),
        ("If the battery is below 20%, enable power-saving mode. My phone is at 12%.", "r_mp_6",
         "Should I enable power-saving mode?",
         "Yes, you should enable power-saving mode because 12% is below 20%.",
         "Wason 1966", ["No need for power saving"]),
        ("If the check engine light comes on, take the car to the mechanic. The light just came on.", "r_mp_7",
         "What should I do with my car?",
         "You should take your car to the mechanic because the check engine light came on.",
         "Wason 1966", ["Ignore the light"]),
        ("If demand exceeds supply, prices will rise. Demand just exceeded supply in the housing market.", "r_mp_8",
         "What will happen to housing prices?",
         "Housing prices will rise because demand has exceeded supply.",
         "Wason 1966", ["Prices will fall"]),
        ("If a dog wags its tail, it's usually happy. My dog is wagging its tail.", "r_mp_9",
         "How is my dog feeling?",
         "Your dog is likely happy because it is wagging its tail.",
         "Wason 1966", ["Your dog is sad"]),
        ("If the alarm goes off, evacuate the building. The alarm is going off.", "r_mp_10",
         "What should I do?",
         "You should evacuate the building because the alarm is going off.",
         "Wason 1966", ["Stay in the building"]),
        ("If you eat that expired yogurt, you'll probably get sick. I ate the expired yogurt.", "r_mp_11",
         "What's likely going to happen to me?",
         "You'll probably get sick because you ate the expired yogurt.",
         "Wason 1966", ["Nothing will happen"]),
        ("If the project is over budget, we need approval from the VP. The project is $10K over budget.", "r_mp_12",
         "Do we need VP approval?",
         "Yes, you need VP approval because the project is over budget.",
         "Wason 1966", ["No approval needed"]),
    ]
    add(dom, "modus_ponens", "If the library book is overdue, there's a fine. My book is two weeks overdue.",
        ref="Wason 1966", group="r_mp_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, reference, forbidden in mp:
        add(dom, "modus_ponens", setup_msg, ref=reference, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "modus_ponens", probe_msg, gold, ref=reference, group=grp,
            forbidden=forbidden)

    # --- modus_tollens (25 prompts: 13 setup + 12 probes) ---
    # If P then Q. Q is false. → P is false.
    mt = [
        ("If it rained last night, the sidewalk would be wet. The sidewalk is dry.", "r_mt_1",
         "Did it rain last night?",
         "No, it did not rain last night because the sidewalk is dry.",
         "Wason 1966", ["It rained last night"]),
        ("If the oven was on, the kitchen would be warm. The kitchen is cold.", "r_mt_2",
         "Was the oven on?",
         "No, the oven was not on because the kitchen is cold.",
         "Wason 1966", ["The oven was on"]),
        ("If Sarah went to the party, she would have posted on Instagram. She didn't post anything.", "r_mt_3",
         "Did Sarah go to the party?",
         "Likely not. If Sarah went, she would have posted, but she didn't.",
         "Wason 1966", ["Sarah went to the party"]),
        ("If the package was delivered, there'd be a notification. I have no notification.", "r_mt_4",
         "Has my package been delivered?",
         "No, your package has not been delivered since you have no delivery notification.",
         "Wason 1966", ["The package was delivered"]),
        ("If the server crashed, the monitoring system would alert us. We received no alert.", "r_mt_5",
         "Did the server crash?",
         "No, the server did not crash because the monitoring system did not send an alert.",
         "Wason 1966", ["The server crashed"]),
        ("If the dog got out, there would be paw prints in the snow. There are no paw prints.", "r_mt_6",
         "Did the dog get out?",
         "No, the dog didn't get out because there are no paw prints in the snow.",
         "Wason 1966", ["The dog got out"]),
        ("If she passed the exam, she would be celebrating. She's not celebrating.", "r_mt_7",
         "Did she pass the exam?",
         "Likely not. If she passed she would be celebrating, but she isn't.",
         "Wason 1966", ["She passed the exam"]),
        ("If there was a gas leak, we'd smell it. There's no smell.", "r_mt_8",
         "Is there a gas leak?",
         "There is likely no gas leak because you can't smell anything.",
         "Wason 1966", ["There is a gas leak"]),
        ("If the meeting was canceled, I'd have an email. I have no such email.", "r_mt_9",
         "Is the meeting still on?",
         "Yes, the meeting is still on because you would have received an email if it were canceled.",
         "Wason 1966", ["The meeting was canceled"]),
        ("If the plant was watered, the soil would be damp. The soil is bone dry.", "r_mt_10",
         "Has someone watered the plant?",
         "No, the plant has not been watered because the soil is bone dry.",
         "Wason 1966", ["The plant was watered"]),
        ("If he studied for the test, he'd feel confident. He feels anxious.", "r_mt_11",
         "Did he study enough?",
         "He likely didn't study enough because he's feeling anxious rather than confident.",
         "Wason 1966", ["He studied well"]),
        ("If the car was serviced, the warning light would be off. The light is still on.", "r_mt_12",
         "Was the car serviced?",
         "The car probably wasn't serviced properly because the warning light is still on.",
         "Wason 1966", ["The car was serviced"]),
    ]
    add(dom, "modus_tollens", "If there was a power outage, the clocks would be reset. The clocks show the correct time.",
        ref="Wason 1966", group="r_mt_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, reference, forbidden in mt:
        add(dom, "modus_tollens", setup_msg, ref=reference, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "modus_tollens", probe_msg, gold, ref=reference, group=grp,
            forbidden=forbidden)

    # --- analogical_reasoning (25 prompts: no setup needed, direct probes) ---
    analogies = [
        ("r_analog_1", "Bird is to nest as bear is to what?",
         "A bear is to a den or cave, just as a bird is to a nest. Both are homes for their respective animals.",
         "Gentner 1983"),
        ("r_analog_2", "Doctor is to hospital as teacher is to what?",
         "A teacher is to a school, just as a doctor is to a hospital. Both are professionals in their respective workplaces.",
         "Gentner 1983"),
        ("r_analog_3", "Paw is to dog as hoof is to what?",
         "A hoof is to a horse, just as a paw is to a dog. Both are the foot structures of their respective animals.",
         "Gentner 1983"),
        ("r_analog_4", "Seed is to flower as egg is to what?",
         "An egg is to a bird or chicken, just as a seed is to a flower. Both are starting points for growth.",
         "Gentner 1983"),
        ("r_analog_5", "Electricity is to wire as water is to what?",
         "Water is to a pipe, just as electricity is to a wire. Both flow through their respective conduits.",
         "Gentner 1983"),
        ("r_analog_6", "Chapter is to book as episode is to what?",
         "An episode is to a TV series or show, just as a chapter is to a book. Both are subdivisions of a larger work.",
         "Gentner 1983"),
        ("r_analog_7", "Steering wheel is to car as handlebars are to what?",
         "Handlebars are to a bicycle, just as a steering wheel is to a car. Both are used to steer their respective vehicles.",
         "Gentner 1983"),
        ("r_analog_8", "Bark is to tree as skin is to what?",
         "Skin is to a human or animal, just as bark is to a tree. Both are outer protective coverings.",
         "Gentner 1983"),
        ("r_analog_9", "Palette is to painter as keyboard is to what?",
         "A keyboard is to a programmer or typist, just as a palette is to a painter. Both are primary tools of their craft.",
         "Gentner 1983"),
        ("r_analog_10", "Spoke is to wheel as ray is to what?",
         "A ray is to the sun, just as a spoke is to a wheel. Both radiate outward from a central point.",
         "Gentner 1983"),
        ("r_analog_11", "Conductor is to orchestra as coach is to what?",
         "A coach is to a team, just as a conductor is to an orchestra. Both lead and direct group performance.",
         "Gentner 1983"),
        ("r_analog_12", "Engine is to car as heart is to what?",
         "A heart is to a body, just as an engine is to a car. Both are the central power sources that keep things running.",
         "Gentner 1983"),
    ]
    # one extra standalone to reach 25
    add(dom, "analogical_reasoning", "Hand is to glove as foot is to what?",
        "A foot is to a shoe or sock, just as a hand is to a glove. Both are coverings for appendages.",
        ref="Gentner 1983", group="r_analog_0")
    for grp, probe_msg, gold, reference in analogies:
        add(dom, "analogical_reasoning", probe_msg, gold, ref=reference, group=grp)
    # extra analogies to reach 25
    bonus_analogies = [
        ("r_analog_13", "Lens is to camera as retina is to what?",
         "A retina is to an eye, just as a lens is to a camera. Both capture and focus images."),
        ("r_analog_14", "Rudder is to ship as tail is to what?",
         "A tail is to an airplane, just as a rudder is to a ship. Both help steer."),
        ("r_analog_15", "Foundation is to house as roots are to what?",
         "Roots are to a tree, just as a foundation is to a house. Both provide structural support."),
        ("r_analog_16", "Brush is to painter as chisel is to what?",
         "A chisel is to a sculptor, just as a brush is to a painter. Both are primary tools of the artist."),
        ("r_analog_17", "Fuel is to rocket as food is to what?",
         "Food is to a human or animal body, just as fuel is to a rocket. Both provide energy."),
        ("r_analog_18", "Antenna is to radio as ear is to what?",
         "An ear is to a person, just as an antenna is to a radio. Both receive signals or sound."),
        ("r_analog_19", "Scale is to fish as feather is to what?",
         "A feather is to a bird, just as a scale is to a fish. Both are external body coverings."),
        ("r_analog_20", "Sheath is to sword as holster is to what?",
         "A holster is to a gun, just as a sheath is to a sword. Both are protective holders for weapons."),
        ("r_analog_21", "Pedal is to bicycle as oar is to what?",
         "An oar is to a boat or rowboat, just as a pedal is to a bicycle. Both provide propulsion."),
        ("r_analog_22", "Frame is to picture as border is to what?",
         "A border is to a country, just as a frame is to a picture. Both define the edges of a space."),
        ("r_analog_23", "Wax is to candle as coal is to what?",
         "Coal is to a furnace or fire, just as wax is to a candle. Both are fuel sources that burn."),
        ("r_analog_24", "Cast is to broken bone as bandage is to what?",
         "A bandage is to a wound, just as a cast is to a broken bone. Both protect and aid healing."),
    ]
    for grp, probe_msg, gold in bonus_analogies:
        add(dom, "analogical_reasoning", probe_msg, gold, ref="Gentner 1983", group=grp)

    # --- transitive_inference (25 prompts: 10 setup + 15 probes) ---
    # A > B, B > C → A > C
    transitive = [
        ("r_trans_1",
         ["Mount Everest is taller than K2. K2 is taller than Kangchenjunga."],
         [("Is Mount Everest taller than Kangchenjunga?",
           "Yes, Mount Everest is taller than Kangchenjunga because Everest is taller than K2 and K2 is taller than Kangchenjunga."),
          ("Which is the shortest of the three mountains?",
           "Kangchenjunga is the shortest of the three since K2 is taller than it and Everest is taller than K2.")]),
        ("r_trans_2",
         ["Alice is older than Bob. Bob is older than Carol."],
         [("Is Alice older than Carol?",
           "Yes, Alice is older than Carol because Alice is older than Bob and Bob is older than Carol."),
          ("Who is the youngest?",
           "Carol is the youngest since Bob is older than Carol and Alice is older than Bob."),
          ("Could Carol be older than Alice?",
           "No, Carol cannot be older than Alice. The chain is Alice > Bob > Carol in age.")]),
        ("r_trans_3",
         ["Diamond is harder than sapphire. Sapphire is harder than quartz."],
         [("Is diamond harder than quartz?",
           "Yes, diamond is harder than quartz because diamond is harder than sapphire and sapphire is harder than quartz."),
          ("Can quartz scratch diamond?",
           "No, quartz cannot scratch diamond because diamond is harder than quartz in the hardness chain.")]),
        ("r_trans_4",
         ["New York is more populous than Chicago. Chicago is more populous than Houston."],
         [("Is New York more populous than Houston?",
           "Yes, New York is more populous than Houston because New York > Chicago > Houston in population."),
          ("Which city has the smallest population of the three?",
           "Houston has the smallest population of the three cities."),
          ("Could Houston be larger than New York?",
           "No, based on the given information, New York is more populous than Chicago which is more populous than Houston.")]),
        ("r_trans_5",
         ["The elephant is heavier than the gorilla. The gorilla is heavier than the wolf."],
         [("Is the elephant heavier than the wolf?",
           "Yes, the elephant is heavier than the wolf because elephant > gorilla > wolf in weight."),
          ("Which animal is the lightest?",
           "The wolf is the lightest of the three animals.")]),
        ("r_trans_6",
         ["Python is easier to learn than C++. C++ is easier to learn than Assembly."],
         [("Is Python easier than Assembly?",
           "Yes, Python is easier to learn than Assembly because Python > C++ > Assembly in ease of learning."),
          ("Which language is the hardest to learn?",
           "Assembly is the hardest to learn of the three languages.")]),
        ("r_trans_7",
         ["The Pacific Ocean is larger than the Atlantic. The Atlantic is larger than the Indian Ocean."],
         [("Is the Pacific larger than the Indian Ocean?",
           "Yes, the Pacific Ocean is larger than the Indian Ocean because Pacific > Atlantic > Indian in size."),
          ("Which ocean is the smallest of the three?",
           "The Indian Ocean is the smallest of the three.")]),
        ("r_trans_8",
         ["Gold is more expensive than silver. Silver is more expensive than copper."],
         [("Is gold more expensive than copper?",
           "Yes, gold is more expensive than copper because gold > silver > copper in price.")]),
    ]
    for grp, setups, probes in transitive:
        for s in setups:
            add(dom, "transitive_inference", s, ref="Piaget 1952", group=grp,
                setup=True, checks=["belief_created"])
        for probe_msg, gold in probes:
            add(dom, "transitive_inference", probe_msg, gold, ref="Piaget 1952", group=grp)

    # --- causal_reasoning (25 prompts: 13 setup + 12 probes) ---
    causal = [
        ("The plant leaves are turning yellow and the soil is very dry.", "r_cause_1",
         "Why are the leaves turning yellow?",
         "The leaves are likely turning yellow because the plant is not getting enough water, as indicated by the very dry soil.",
         "Tversky & Kahneman 1974"),
        ("Sales dropped 30% right after we raised prices by 25%.", "r_cause_2",
         "What likely caused the sales drop?",
         "The 25% price increase likely caused the 30% sales drop, as customers were deterred by the higher prices.",
         "Tversky & Kahneman 1974"),
        ("The road is icy and three cars have slid off this curve today.", "r_cause_3",
         "Why are cars sliding off the curve?",
         "Cars are sliding off the curve because the road is icy, reducing traction at the turn.",
         "Tversky & Kahneman 1974"),
        ("My son ate a whole box of candy and now has a stomachache.", "r_cause_4",
         "What caused my son's stomachache?",
         "Your son's stomachache was likely caused by eating the whole box of candy.",
         "Tversky & Kahneman 1974"),
        ("Since I started using a standing desk, my back pain has decreased significantly.", "r_cause_5",
         "What seems to be helping my back pain?",
         "Using a standing desk seems to be helping reduce your back pain.",
         "Tversky & Kahneman 1974"),
        ("The wifi went out right when the thunderstorm started.", "r_cause_6",
         "What probably caused the wifi outage?",
         "The thunderstorm likely caused the wifi outage, possibly through a power disruption or interference.",
         "Tversky & Kahneman 1974"),
        ("Employee satisfaction surveys improved after we introduced flexible work hours.", "r_cause_7",
         "What improved employee satisfaction?",
         "The introduction of flexible work hours likely improved employee satisfaction.",
         "Tversky & Kahneman 1974"),
        ("The fish in the lake started dying after the factory upstream began discharging waste.", "r_cause_8",
         "Why are the fish dying?",
         "The fish are likely dying because of the factory's waste discharge contaminating the lake water.",
         "Tversky & Kahneman 1974"),
        ("My toddler skipped his nap and now he's throwing a tantrum.", "r_cause_9",
         "Why is my toddler having a tantrum?",
         "Your toddler is likely having a tantrum because he skipped his nap and is overtired.",
         "Tversky & Kahneman 1974"),
        ("The tomato plants grew much taller after I added compost to the soil.", "r_cause_10",
         "What made the tomato plants grow taller?",
         "Adding compost to the soil likely provided nutrients that made the tomato plants grow taller.",
         "Tversky & Kahneman 1974"),
        ("Traffic on the highway doubled after the bridge on the alternate route closed.", "r_cause_11",
         "Why did traffic increase?",
         "Traffic increased because the alternate route's bridge closed, forcing more drivers onto the highway.",
         "Tversky & Kahneman 1974"),
        ("My laptop battery started draining fast after the latest software update.", "r_cause_12",
         "What's causing the battery drain?",
         "The latest software update is likely causing the battery to drain faster.",
         "Tversky & Kahneman 1974"),
    ]
    add(dom, "causal_reasoning", "The office building lost power and all the computers shut down.",
        ref="Tversky & Kahneman 1974", group="r_cause_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, reference in causal:
        add(dom, "causal_reasoning", setup_msg, ref=reference, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "causal_reasoning", probe_msg, gold, ref=reference, group=grp)

# ====================================================================
# DOMAIN 7: SOCIAL COGNITION (125 prompts)
# Premack & Woodruff 1978, Wimmer & Perner 1983, Kohlberg 1958
# ====================================================================
def _social_cognition(add):
    dom = "social_cognition"

    # --- false_belief (25 prompts: 10 setup + 15 probes) ---
    # Sally-Anne variants: person A puts X in location 1. Person B moves X
    # to location 2 while A is away. Where does A look?
    false_belief = [
        ("sc_fb_1",
         ["Sarah put her lunch in the fridge before the meeting.",
          "While Sarah was in the meeting, Mike moved her lunch to the cabinet."],
         [("Where will Sarah look for her lunch when she gets back?",
           "Sarah will look in the fridge because she doesn't know Mike moved her lunch to the cabinet.",
           ["Sarah will look in the cabinet"]),
          ("Does Sarah know her lunch was moved?",
           "No, Sarah does not know her lunch was moved because she was in the meeting when Mike moved it.",
           []),
          ("Where is the lunch actually?",
           "The lunch is actually in the cabinet where Mike moved it.",
           ["The lunch is in the fridge"])]),
        ("sc_fb_2",
         ["Tom hid a toy car under his pillow before going to school.",
          "His mom found it while cleaning and put it in the toy box."],
         [("Where will Tom look for his toy car when he gets home?",
           "Tom will look under his pillow because he doesn't know his mom moved it to the toy box.",
           ["Tom will look in the toy box"]),
          ("Does Tom know the toy was moved?",
           "No, Tom doesn't know the toy was moved. He was at school when his mom cleaned.",
           []),
          ("Where is the toy car now?",
           "The toy car is now in the toy box where Tom's mom put it.",
           ["The toy is under the pillow"])]),
        ("sc_fb_3",
         ["Lisa left her umbrella by the front door before work.",
          "Her roommate moved it to the hall closet while Lisa was at work."],
         [("Where will Lisa look for her umbrella?",
           "Lisa will look by the front door because she doesn't know her roommate moved it to the closet.",
           ["Lisa will check the closet"]),
          ("Why won't Lisa check the closet?",
           "Lisa won't check the closet because she still believes the umbrella is by the front door where she left it.",
           []),
          ("Where is the umbrella really?",
           "The umbrella is in the hall closet where Lisa's roommate moved it.",
           [])]),
        ("sc_fb_4",
         ["Dave put the leftover pizza in the oven to keep it warm while he showered.",
          "His wife ate the pizza and cleaned up while Dave was in the shower."],
         [("Where does Dave think the pizza is?",
           "Dave thinks the pizza is still in the oven because he doesn't know his wife ate it.",
           ["Dave knows the pizza is gone"]),
          ("Will Dave be surprised when he checks the oven?",
           "Yes, Dave will be surprised because he expects to find pizza in the oven but it's gone.",
           []),
          ("Does Dave know the pizza was eaten?",
           "No, Dave does not know the pizza was eaten. He was in the shower.",
           [])]),
    ]
    for grp, setups, probes in false_belief:
        for s in setups:
            add(dom, "false_belief", s, ref="Wimmer & Perner 1983", group=grp,
                setup=True, checks=["belief_created"])
        for probe_msg, gold, forbidden in probes:
            add(dom, "false_belief", probe_msg, gold, ref="Wimmer & Perner 1983",
                group=grp, forbidden=forbidden)
    # extra standalone to reach 25
    add(dom, "false_belief", "Jenny placed her book on the kitchen table and went outside.",
        ref="Wimmer & Perner 1983", group="sc_fb_5", setup=True, checks=["belief_created"])
    add(dom, "false_belief", "Her dad moved the book to the bookshelf while Jenny was outside.",
        ref="Wimmer & Perner 1983", group="sc_fb_5", setup=True, checks=["belief_created"])
    add(dom, "false_belief", "Where will Jenny look for her book when she comes inside?",
        "Jenny will look on the kitchen table because she doesn't know her dad moved it to the bookshelf.",
        ref="Wimmer & Perner 1983", group="sc_fb_5",
        forbidden=["Jenny will look on the bookshelf"])
    add(dom, "false_belief", "Does Jenny know the book was moved?",
        "No, Jenny does not know the book was moved because she was outside.",
        ref="Wimmer & Perner 1983", group="sc_fb_5")
    add(dom, "false_belief", "Where is the book actually?",
        "The book is on the bookshelf where Jenny's dad moved it.",
        ref="Wimmer & Perner 1983", group="sc_fb_5")

    # --- perspective_taking (25 prompts: 13 setup + 12 probes) ---
    perspective = [
        ("My 5-year-old asked why the moon follows our car.", "sc_persp_1",
         "How should I explain this to a 5-year-old?",
         "You could explain that the moon is so far away that it seems to follow the car, like how distant mountains barely move when you drive. It's an illusion because of how far away it is.",
         "Premack & Woodruff 1978"),
        ("My elderly mother is confused by the new TV remote with 50 buttons.", "sc_persp_2",
         "How might she be feeling?",
         "Your mother is likely feeling frustrated and overwhelmed by the complexity of the remote. Too many buttons can be intimidating for someone not used to modern technology.",
         "Premack & Woodruff 1978"),
        ("A blind colleague asked me to describe what the sunset looks like.", "sc_persp_3",
         "How would I describe a sunset to a blind person?",
         "You could describe a sunset using other senses: the warmth fading from your skin, like a warm blanket slowly being pulled away. The colors transition like going from the heat of a cup of tea to the coolness of water.",
         "Premack & Woodruff 1978"),
        ("My friend who just moved from tropical Brazil is experiencing their first winter in Minnesota.", "sc_persp_4",
         "What challenges might they face?",
         "They would likely struggle with the extreme cold, need to learn about winter clothing and driving on ice, and could experience seasonal depression from the short days and lack of sunlight.",
         "Premack & Woodruff 1978"),
        ("A new immigrant colleague doesn't understand why everyone is excited about Thanksgiving.", "sc_persp_5",
         "Why might they feel confused?",
         "Thanksgiving is a cultural tradition specific to the US and Canada, so someone without that cultural background wouldn't have emotional associations with the holiday. The excitement would seem arbitrary without understanding the tradition.",
         "Premack & Woodruff 1978"),
        ("My teenager rolled their eyes when I said 'that's lit.'", "sc_persp_6",
         "Why did my teenager react that way?",
         "Your teenager likely found it embarrassing or cringeworthy for a parent to use youth slang. Teens often feel their slang loses its coolness when adults adopt it.",
         "Premack & Woodruff 1978"),
        ("A colleague who grew up in poverty seems uncomfortable at the fancy company dinner.", "sc_persp_7",
         "Why might they feel uncomfortable?",
         "They may feel out of place in an environment that signals wealth and social status they didn't grow up with. Unfamiliar etiquette, expensive settings, and class differences can trigger feelings of inadequacy.",
         "Premack & Woodruff 1978"),
        ("My dog cowers when I pick up a broom, even though I've never hit him.", "sc_persp_8",
         "Why might my dog react that way?",
         "Your dog may have had a negative experience with raised objects before you adopted him, possibly from a previous owner. This is a learned fear response to something that resembles a past threat.",
         "Premack & Woodruff 1978"),
        ("A Japanese business partner seemed offended when I opened their gift immediately.", "sc_persp_9",
         "Why might they be offended?",
         "In Japanese culture, it is customary to not open gifts in front of the giver, as it can seem impatient or greedy. Opening it immediately violated their cultural expectation of politeness.",
         "Premack & Woodruff 1978"),
        ("My 3-year-old is crying because her imaginary friend wasn't invited to a real birthday party.", "sc_persp_10",
         "How should I handle this situation?",
         "At age 3, imaginary friends feel very real. Dismissing the concern would hurt her feelings. You could validate her emotions and suggest that her imaginary friend could come along with her.",
         "Premack & Woodruff 1978"),
        ("My coworker always eats lunch alone and never joins team outings.", "sc_persp_11",
         "What might explain this behavior?",
         "They might be introverted and need solitude to recharge, or have social anxiety that makes group situations stressful. It's also possible they have personal reasons like dietary restrictions or obligations.",
         "Premack & Woodruff 1978"),
        ("A student burst into tears when getting a B+ on their test.", "sc_persp_12",
         "Why would a B+ make someone cry?",
         "The student might have extremely high standards or pressure from parents. For a perfectionist, a B+ can feel like failure. They might also need a higher grade to maintain a scholarship or get into a desired program.",
         "Premack & Woodruff 1978"),
    ]
    add(dom, "perspective_taking", "My neighbor who recently lost his wife seems angry when I offer help.",
        ref="Premack & Woodruff 1978", group="sc_persp_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, reference in perspective:
        add(dom, "perspective_taking", setup_msg, ref=reference, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "perspective_taking", probe_msg, gold, ref=reference, group=grp)

    # --- emotional_inference (25 prompts: 13 setup + 12 probes) ---
    emotions = [
        ("My colleague just found out she didn't get the promotion she's been working toward for 2 years.", "sc_emo_1",
         "How is my colleague likely feeling?",
         "Your colleague is likely feeling deeply disappointed, frustrated, and possibly angry or demoralized after two years of effort without recognition.",
         "Baron-Cohen et al. 1997"),
        ("My friend keeps checking his phone every 30 seconds while waiting for medical test results.", "sc_emo_2",
         "What emotional state is my friend in?",
         "Your friend is anxious and nervous about the medical test results. The compulsive phone checking indicates high anxiety and anticipation.",
         "Baron-Cohen et al. 1997"),
        ("My daughter came home from school and went straight to her room without saying hi.", "sc_emo_3",
         "What might my daughter be feeling?",
         "Your daughter might be upset, embarrassed, or had a bad day at school. Withdrawing and avoiding interaction often indicates emotional distress in children.",
         "Baron-Cohen et al. 1997"),
        ("My husband has been unusually quiet and keeps sighing heavily since he talked to his boss.", "sc_emo_4",
         "What might be going on with my husband?",
         "Your husband is likely worried or stressed about something that happened with his boss. The sighing and quietness suggest he received bad news or is under pressure.",
         "Baron-Cohen et al. 1997"),
        ("My coworker slammed her laptop shut and said 'great, just great' in a flat tone.", "sc_emo_5",
         "Is my coworker actually saying things are great?",
         "No, your coworker is being sarcastic. She is frustrated or angry about something. Slamming the laptop and the flat tone indicate the opposite of what the words say.",
         "Baron-Cohen et al. 1997"),
        ("After winning the championship, the coach stood silently with tears streaming down his face.", "sc_emo_6",
         "Why is the coach crying if his team won?",
         "The coach is crying tears of joy and relief. Overwhelming positive emotions, especially after long effort, can express themselves as tears.",
         "Baron-Cohen et al. 1997"),
        ("My friend smiled and said 'I'm fine' but her voice was shaking.", "sc_emo_7",
         "Is my friend really fine?",
         "No, your friend is likely not fine. The shaking voice contradicts the smile and words, suggesting she is suppressing distress and putting on a brave face.",
         "Baron-Cohen et al. 1997"),
        ("My toddler threw his plate of food on the floor and started screaming.", "sc_emo_8",
         "What's causing this behavior?",
         "Your toddler may be frustrated, overtired, or overwhelmed. Toddlers lack the verbal ability to express complex emotions, so they act out physically.",
         "Baron-Cohen et al. 1997"),
        ("My elderly father keeps giving away his possessions to family members.", "sc_emo_9",
         "Why might my father be doing this?",
         "Your father may be preparing for the end of his life, wanting to see his loved ones enjoy his possessions while he's still here. This behavior often reflects acceptance of mortality and a desire to leave a meaningful legacy.",
         "Baron-Cohen et al. 1997"),
        ("My friend who just went through a divorce is suddenly taking up extreme sports.", "sc_emo_10",
         "Why this sudden interest in extreme sports?",
         "Your friend may be coping with the emotional pain of divorce by seeking adrenaline and distraction. Extreme activities can provide a sense of control and excitement during a time of emotional upheaval.",
         "Baron-Cohen et al. 1997"),
        ("My colleague laughed very loudly at a joke that wasn't funny during the awkward meeting.", "sc_emo_11",
         "Why did my colleague laugh so much?",
         "Your colleague was likely trying to break the tension of the awkward meeting. The exaggerated laughter was a social strategy to ease discomfort, not a genuine reaction to humor.",
         "Baron-Cohen et al. 1997"),
        ("My neighbor keeps her curtains closed and hasn't been seen outside in weeks.", "sc_emo_12",
         "What might be going on with my neighbor?",
         "Your neighbor may be experiencing depression, grief, or social withdrawal. Extended isolation and closing off from the outside world are often signs of emotional distress.",
         "Baron-Cohen et al. 1997"),
    ]
    add(dom, "emotional_inference", "My student who normally participates enthusiastically has been silent for three classes in a row.",
        ref="Baron-Cohen et al. 1997", group="sc_emo_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, reference in emotions:
        add(dom, "emotional_inference", setup_msg, ref=reference, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "emotional_inference", probe_msg, gold, ref=reference, group=grp)

    # --- moral_reasoning (25 prompts: direct probes, no setup) ---
    morals = [
        ("sc_mor_1", "A train is heading toward 5 people. You can pull a lever to divert it to a track with 1 person. Should you pull the lever?",
         "This is a classic trolley problem. The utilitarian argument says yes, pull the lever to save 5 lives at the cost of 1. The deontological argument says deliberately diverting the train makes you actively responsible for the one death. Most ethical frameworks consider saving more lives as the right choice, but reasonable people disagree.",
         "Kohlberg 1958"),
        ("sc_mor_2", "Your best friend confides that they cheated on their professional certification exam. Should you report it?",
         "This creates a conflict between loyalty to your friend and professional integrity. Reporting protects the public and upholds standards, while not reporting could put others at risk if your friend is unqualified. The ethical obligation leans toward reporting, but the personal cost is significant.",
         "Kohlberg 1958"),
        ("sc_mor_3", "You find a wallet with $500 cash and an ID. The owner is wealthy. Should you return all the money?",
         "Yes, you should return all the money. The owner's wealth doesn't change your ethical obligation. Taking the money would be theft regardless of the owner's financial status.",
         "Kohlberg 1958"),
        ("sc_mor_4", "A parent steals bread to feed their starving children. Is this morally justified?",
         "Many ethical frameworks would consider this justified. While theft is generally wrong, the immediate threat to the children's lives creates a moral imperative that overrides property rights. This is a classic example of necessity vs. law.",
         "Kohlberg 1958"),
        ("sc_mor_5", "Your company's product has a minor safety flaw that hasn't caused injuries yet. Fixing it would cost $2 million. Should you fix it or wait?",
         "You should fix it proactively. Waiting until someone gets hurt is ethically and legally riskier. Companies have a duty of care to their customers, and knowingly selling a product with a safety flaw, even minor, is negligent.",
         "Kohlberg 1958"),
        ("sc_mor_6", "You witness a coworker being bullied by your boss. Speaking up might cost you your job. What should you do?",
         "Speaking up is the ethical choice, though it comes with personal risk. You could document the behavior, report to HR, or support your coworker privately. Silence enables the bullying to continue.",
         "Kohlberg 1958"),
        ("sc_mor_7", "A self-driving car must choose between hitting an elderly person or a child. How should it be programmed?",
         "This is an impossible choice with no clean ethical answer. Utilitarian logic might compare expected life years, but assigning different values to human lives based on age is deeply problematic. Most ethicists argue the car should minimize total harm without discriminating based on personal characteristics.",
         "Kohlberg 1958"),
        ("sc_mor_8", "You discover your employer is illegally dumping waste but reporting it would shut down the factory and cost 200 people their jobs.",
         "The environmental and health damage from illegal dumping affects potentially far more people long-term. Reporting is the right choice, though the job losses are a serious consequence. Whistleblower protections exist for this reason.",
         "Kohlberg 1958"),
        ("sc_mor_9", "Is it ethical to lie to spare someone's feelings? For example, telling a friend their painting is good when it's not.",
         "This involves a tension between honesty and compassion. A kind lie in a low-stakes situation (casual compliment) differs from a consequential one (professional feedback). Constructive honesty that acknowledges effort while gently noting areas for growth is often the most ethical path.",
         "Kohlberg 1958"),
        ("sc_mor_10", "A doctor has 5 dying patients who each need a different organ. A healthy person walks in. Should the doctor harvest their organs?",
         "No. Despite saving 5 lives, killing an innocent person violates fundamental medical ethics and the principle that people cannot be used merely as means to others' ends. This is a key distinction between utilitarian outcomes and moral constraints.",
         "Kohlberg 1958"),
        ("sc_mor_11", "You promised your child you'd attend their school play, but your boss demands you work late on an urgent project.",
         "Breaking a promise to your child has lasting emotional impact and teaches them they can't rely on your word. If possible, negotiate with your boss for compromise. If forced to choose, the child's trust and emotional well-being should weigh heavily in the decision.",
         "Kohlberg 1958"),
        ("sc_mor_12", "Is it wrong to use ad-blockers when content creators rely on ad revenue?",
         "This involves weighing personal autonomy (controlling your own browsing experience) against supporting creators. While not theft, ad-blocking does reduce creator revenue. Many consider it ethical if you support creators through other means, but using content while blocking its funding source raises fairness questions.",
         "Kohlberg 1958"),
    ]
    add(dom, "moral_reasoning", "Is it ethical to eat animals when plant-based alternatives exist?",
        "This involves tensions between cultural tradition, personal autonomy, animal welfare, environmental impact, and nutritional needs. Ethical frameworks differ: utilitarianism emphasizes reducing suffering, while cultural and personal autonomy arguments support individual choice.",
        ref="Kohlberg 1958", group="sc_mor_0")
    for grp, probe_msg, gold, reference in morals:
        add(dom, "moral_reasoning", probe_msg, gold, ref=reference, group=grp)
    # extra moral reasoning probes to reach 25
    extra_morals = [
        ("sc_mor_13", "Is it ethical to keep animals in zoos for education and conservation?",
         "Zoos serve conservation and education but confine animals. The ethics depend on welfare standards, whether species truly benefit from conservation programs, and whether educational goals justify captivity.",
         "Kohlberg 1958"),
        ("sc_mor_14", "Should wealthy nations accept unlimited refugees during humanitarian crises?",
         "There's a moral obligation to help those in danger balanced against practical capacity. Most ethical frameworks support a strong duty to assist, but disagree on how far that duty extends when resources are strained.",
         "Kohlberg 1958"),
        ("sc_mor_15", "Is it wrong to read someone's diary if you suspect they're in danger?",
         "Normally reading someone's diary violates their privacy and trust. However, if you genuinely believe they are in danger (suicidal, being abused), the ethical calculus changes — protecting someone's life may outweigh privacy.",
         "Kohlberg 1958"),
        ("sc_mor_16", "Should parents monitor their teenager's social media activity?",
         "This balances a parent's duty to protect their child against the teen's developing autonomy and privacy. Age-appropriate monitoring that decreases as the teen demonstrates responsibility is often considered the ethical middle ground.",
         "Kohlberg 1958"),
        ("sc_mor_17", "Is it ethical to use data from unethical experiments if it could save lives today?",
         "Using such data benefits current patients but risks normalizing unethical research. Many argue the data should be used if it can save lives, but with full acknowledgment of how it was obtained and honoring the victims.",
         "Kohlberg 1958"),
        ("sc_mor_18", "You find out a close friend is cheating on their partner. Should you tell the partner?",
         "This creates a conflict between loyalty to your friend and honesty to their partner. Most ethical frameworks say the partner has a right to know, but the method matters: encouraging your friend to come clean first is often the recommended approach.",
         "Kohlberg 1958"),
        ("sc_mor_19", "Is it ethical for a teacher to give extra help to struggling students while high-performing students receive less attention?",
         "Equity versus equality: treating students differently based on need aims for equal outcomes. Most educational ethics support differentiated attention, though high-performing students also deserve appropriate challenge.",
         "Kohlberg 1958"),
        ("sc_mor_20", "Should autonomous weapons be allowed in warfare if they reduce civilian casualties?",
         "Even if they reduce casualties, delegating life-or-death decisions to machines raises deep ethical concerns about accountability, the dignity of human judgment in warfare, and the risk of lowering the threshold for conflict.",
         "Kohlberg 1958"),
        ("sc_mor_21", "Is it ethical to genetically modify human embryos to prevent hereditary diseases?",
         "Preventing disease is a compelling goal, but germline modification raises concerns about consent of the future person, unforeseen consequences, and the slippery slope toward enhancement rather than treatment.",
         "Kohlberg 1958"),
        ("sc_mor_22", "A homeless person asks for money. You suspect they'll use it for alcohol. Should you give it?",
         "This involves respecting another person's autonomy versus paternalism. You can't know their intent, and the discomfort of uncertainty doesn't override their agency. Offering food or resources is a common compromise.",
         "Kohlberg 1958"),
        ("sc_mor_23", "Should social media companies censor misinformation even if it restricts free speech?",
         "This pits public safety against free expression. Unchecked misinformation can cause real harm, but centralized censorship creates power imbalances. Transparency about moderation policies is widely seen as essential.",
         "Kohlberg 1958"),
        ("sc_mor_24", "You accidentally damage a parked car but nobody saw. Should you leave a note?",
         "Yes, you should leave a note with your contact information. Even though nobody saw, honesty and taking responsibility are fundamental ethical obligations. Driving away is essentially making the victim pay for your mistake.",
         "Kohlberg 1958"),
    ]
    for grp, probe_msg, gold, reference in extra_morals:
        add(dom, "moral_reasoning", probe_msg, gold, ref=reference, group=grp)

    # --- intention_attribution (25 prompts: 13 setup + 12 probes) ---
    intentions = [
        ("A child broke a plate while trying to surprise their mom by washing the dishes.", "sc_int_1",
         "Should the child be punished for breaking the plate?",
         "The child's intention was to help and surprise their mom, not to break the plate. The accident happened during a well-intentioned act, so punishment would be disproportionate. Acknowledgment and guidance on being careful would be more appropriate.",
         "Premack & Woodruff 1978"),
        ("My coworker took credit for my idea in the meeting but later told the boss it was mine.", "sc_int_2",
         "What was my coworker's intent?",
         "Your coworker may have initially gotten caught up in the moment or wanted to present the idea effectively, then corrected the attribution. The fact that they told the boss gives credit where due suggests they weren't trying to permanently steal credit.",
         "Premack & Woodruff 1978"),
        ("A stranger is following me down the street but they're looking at their phone and seem distracted.", "sc_int_3",
         "Is the stranger following me intentionally?",
         "Most likely not. They appear distracted by their phone and are probably just walking in the same direction. True following would involve more focused attention on you rather than their phone.",
         "Premack & Woodruff 1978"),
        ("My friend cancelled our dinner plans for the third time this month.", "sc_int_4",
         "Is my friend avoiding me?",
         "Three cancellations could indicate avoidance, but could also mean they're genuinely busy, struggling with something personal, or dealing with scheduling conflicts. The intent depends on whether they suggest rescheduling and how they communicate about it.",
         "Premack & Woodruff 1978"),
        ("A car honked at me right after the light turned green.", "sc_int_5",
         "Was the driver being aggressive?",
         "A quick honk right after the light changes is usually a reminder that the light has changed, not aggression. Aggressive honking tends to be longer, repeated, and accompanied by other behaviors like tailgating.",
         "Premack & Woodruff 1978"),
        ("My neighbor put up a tall fence right after we got a dog.", "sc_int_6",
         "Was the fence because of our dog?",
         "The timing suggests a possible connection — they may wanting to keep the dog out of their yard. However, they may also have been planning the fence anyway for privacy or other reasons. The correlation doesn't prove causation.",
         "Premack & Woodruff 1978"),
        ("A cashier gave me extra change and then smiled.", "sc_int_7",
         "Did the cashier intentionally give me extra change?",
         "Most likely it was an honest mistake. Cashiers handle many transactions and errors happen. The smile was probably standard customer service friendliness, not a knowing signal about the extra change.",
         "Premack & Woodruff 1978"),
        ("My boss scheduled a one-on-one meeting with me and marked it 'confidential.'", "sc_int_8",
         "Should I be worried about this meeting?",
         "A confidential meeting could be about many things: a raise, a new project, organizational changes, feedback, or yes, potentially a concern. The confidential label protects sensitive information but doesn't necessarily indicate bad news.",
         "Premack & Woodruff 1978"),
        ("A student always sits in the back row and never makes eye contact with the professor.", "sc_int_9",
         "Is the student disinterested?",
         "Not necessarily. They could be shy or introverted, have social anxiety, come from a culture where direct eye contact with authority is considered disrespectful, or simply prefer the back row. Avoidance of eye contact doesn't automatically mean disinterest.",
         "Premack & Woodruff 1978"),
        ("Someone left an anonymous positive note on my desk at work.", "sc_int_10",
         "Why would someone leave an anonymous note?",
         "They likely wanted to brighten your day without seeking credit or feeling awkward about it. Anonymity removes social pressure and suggests genuine kindness rather than wanting something in return.",
         "Premack & Woodruff 1978"),
        ("My friend always changes the subject when I mention my promotion.", "sc_int_11",
         "Why does my friend change the subject?",
         "Your friend might be feeling jealous, insecure, or uncomfortable about their own career situation. Changing the subject is a common avoidance behavior when a topic triggers difficult emotions.",
         "Premack & Woodruff 1978"),
        ("A stranger returned my dropped wallet and refused the reward I offered.", "sc_int_12",
         "Why did they refuse the reward?",
         "They likely consider returning lost property as simply the right thing to do and would feel uncomfortable accepting money for basic honesty. Refusing the reward suggests intrinsic moral motivation rather than external incentive.",
         "Premack & Woodruff 1978"),
    ]
    add(dom, "intention_attribution", "A colleague always proofreads my emails before I send them to clients.",
        ref="Premack & Woodruff 1978", group="sc_int_0", setup=True, checks=["belief_created"])
    for setup_msg, grp, probe_msg, gold, reference in intentions:
        add(dom, "intention_attribution", setup_msg, ref=reference, group=grp,
            setup=True, checks=["belief_created"])
        add(dom, "intention_attribution", probe_msg, gold, ref=reference, group=grp)

# ====================================================================
# DOMAIN 8: SELF-CORRECTION + BELIEF LIFECYCLE (125 prompts)
# AGM 1985 — revision, verification, minimal change, decay, dormancy
# ====================================================================
def _self_correction(add):
    dom = "self_correction"

    # --- belief_revision (25 prompts: 10 setup + 15 probes) ---
    # Tell a fact, then contradict it, then verify the correction took hold
    revisions = [
        ("sc_rev_1",
         ["The project launch date is March 15th.",
          "Actually, the project launch has been pushed back to April 1st."],
         [("When is the project launch?", "The project launch is April 1st.",
           ["The project launch is March 15th"]),
          ("Has the launch date changed from the original plan?",
           "Yes, the launch date changed from March 15th to April 1st.",
           []),
          ("What was the original launch date?",
           "The original launch date was March 15th, but it has been moved to April 1st.",
           [])]),
        ("sc_rev_2",
         ["The restaurant reservation is for 6 people.",
          "We added two more guests. The reservation is now for 8 people."],
         [("How many people is the reservation for?", "The reservation is for 8 people.",
           ["The reservation is for 6 people"]),
          ("Did the reservation size change?",
           "Yes, it increased from 6 to 8 people.",
           []),
          ("How many guests were added?",
           "Two guests were added, bringing the total from 6 to 8.",
           [])]),
        ("sc_rev_3",
         ["I thought the meeting was at 2pm.",
          "I was wrong. The meeting is actually at 3pm."],
         [("What time is the meeting?", "The meeting is at 3pm.",
           ["The meeting is at 2pm"]),
          ("Was there a correction about the meeting time?",
           "Yes, you initially thought it was at 2pm but corrected it to 3pm.",
           []),
          ("Should I show up at 2pm?",
           "No, the meeting is at 3pm not 2pm. You corrected the time.",
           [])]),
    ]
    for grp, setups, probes in revisions:
        for idx, s in enumerate(setups):
            checks = ["belief_created"] if idx == 0 else ["tension_increased"]
            add(dom, "belief_revision", s, ref="AGM 1985", group=grp,
                setup=True, checks=checks)
        for probe_msg, gold, forbidden in probes:
            add(dom, "belief_revision", probe_msg, gold, ref="AGM 1985",
                group=grp, forbidden=forbidden)
    # one more pair for count
    add(dom, "belief_revision", "The budget for Q3 was $50,000.",
        ref="AGM 1985", group="sc_rev_4", setup=True, checks=["belief_created"])
    add(dom, "belief_revision", "Correction: the Q3 budget is actually $65,000.",
        ref="AGM 1985", group="sc_rev_4", setup=True, checks=["tension_increased"])
    add(dom, "belief_revision", "What is the Q3 budget?",
        "The Q3 budget is $65,000.", ref="AGM 1985", group="sc_rev_4",
        forbidden=["Q3 budget is $50,000"])
    # extra belief_revision sequences (+7 to reach 25)
    rev_extras = [
        ("sc_rev_5", "The dentist said I need two fillings.", "Actually, after a second look the dentist says I only need one filling.",
         "How many fillings do I need?", "You need one filling. The dentist revised from two to one after a second examination.",
         ["You need two fillings"]),
        ("sc_rev_6", "Our flight is with American Airlines.", "Oops, I mixed it up. Our flight is with Delta.",
         "Which airline are we flying?", "You are flying with Delta.",
         ["flying with American Airlines"]),
    ]
    for grp, s1, s2, q, gold, forb in rev_extras:
        add(dom, "belief_revision", s1, ref="AGM 1985", group=grp, setup=True, checks=["belief_created"])
        add(dom, "belief_revision", s2, ref="AGM 1985", group=grp, setup=True, checks=["tension_increased"])
        add(dom, "belief_revision", q, gold, ref="AGM 1985", group=grp, forbidden=forb)
    add(dom, "belief_revision", "The venue deposit is $500.",
        ref="AGM 1985", group="sc_rev_7", setup=True, checks=["belief_created"])

    # --- revision_verification (20 prompts: 8 setup + 12 probes) ---
    # After correction, use the corrected fact in a novel context
    verifications = [
        ("sc_ver_1",
         ["My car gets 30 miles per gallon.",
          "I had the car tuned up and now it gets 35 miles per gallon."],
         [("If I drive 350 miles, how many gallons will I use?",
           "At 35 miles per gallon, a 350-mile trip would use 10 gallons of fuel.",
           ["At 30 miles per gallon"]),
          ("Has my fuel efficiency improved?",
           "Yes, your fuel efficiency improved from 30 mpg to 35 mpg after the tune-up.",
           []),
          ("What's my current mpg?",
           "Your car currently gets 35 miles per gallon.",
           ["Your car gets 30 miles per gallon"])]),
        ("sc_ver_2",
         ["My rent is $1,200 per month.",
          "My landlord just raised the rent to $1,400 per month."],
         [("How much will I pay in rent over the next 6 months?",
           "At $1,400 per month, you'll pay $8,400 in rent over the next 6 months.",
           ["$7,200", "$1,200 per month"]),
          ("What was my rent increase?",
           "Your rent increased by $200, from $1,200 to $1,400 per month.",
           []),
          ("What do I pay in rent now?",
           "You pay $1,400 per month in rent.",
           ["You pay $1,200"])]),
        ("sc_ver_3",
         ["Our team has 8 members.",
          "We just hired 3 new people, bringing the team to 11."],
         [("If we split into groups of 3, how many full groups can we make?",
           "With 11 team members, you can make 3 full groups of 3 with 2 people remaining.",
           ["8 members", "2 full groups"]),
          ("How many new hires did we add?",
           "You added 3 new hires, growing the team from 8 to 11 members.",
           []),
          ("How big is the team now?",
           "Your team now has 11 members.",
           ["Your team has 8 members"])]),
    ]
    for grp, setups, probes in verifications:
        for idx, s in enumerate(setups):
            checks = ["belief_created"] if idx == 0 else ["tension_increased"]
            add(dom, "revision_verification", s, ref="AGM 1985", group=grp,
                setup=True, checks=checks)
        for probe_msg, gold, forbidden in probes:
            add(dom, "revision_verification", probe_msg, gold, ref="AGM 1985",
                group=grp, forbidden=forbidden)
    # extra probes
    add(dom, "revision_verification", "Our internet speed was 100 Mbps.",
        ref="AGM 1985", group="sc_ver_4", setup=True, checks=["belief_created"])
    add(dom, "revision_verification", "We upgraded to 500 Mbps internet.",
        ref="AGM 1985", group="sc_ver_4", setup=True, checks=["tension_increased"])
    add(dom, "revision_verification", "What is our current internet speed?",
        "Your internet speed is now 500 Mbps.", ref="AGM 1985", group="sc_ver_4",
        forbidden=["Your internet speed is 100 Mbps"])
    add(dom, "revision_verification", "The gym was open 6am-10pm.",
        ref="AGM 1985", group="sc_ver_5", setup=True, checks=["belief_created"])
    add(dom, "revision_verification", "Gym extended hours to 5am-11pm.",
        ref="AGM 1985", group="sc_ver_5", setup=True, checks=["tension_increased"])

    # --- minimal_change (20 prompts: 8 setup + 12 probes) ---
    # Correct one fact in a set, verify unrelated facts remain intact
    minimal = [
        ("sc_min_1",
         ["I have three pets: a dog named Rex, a cat named Luna, and a fish named Bubbles.",
          "Actually, the cat's name is not Luna, it's Cleo."],
         [("What are my pets' names?",
           "You have a dog named Rex, a cat named Cleo, and a fish named Bubbles.",
           ["a cat named Luna"]),
          ("Did anything change about my dog?",
           "No, your dog is still named Rex. Only the cat's name was corrected from Luna to Cleo.",
           []),
          ("What about my fish?",
           "Your fish is still named Bubbles. Only the cat's name changed.",
           [])]),
        ("sc_min_2",
         ["The meeting is Tuesday at 2pm in Room 200 with the marketing team.",
          "Correction: the room changed to Room 305. Everything else stays the same."],
         [("What room is the meeting in?",
           "The meeting is in Room 305.",
           ["Room 200"]),
          ("Is the meeting still on Tuesday?",
           "Yes, the meeting is still on Tuesday at 2pm. Only the room changed to 305.",
           []),
          ("Who is the meeting with?",
           "The meeting is with the marketing team. Only the room changed.",
           [])]),
        ("sc_min_3",
         ["For the trip I need: passport, sunscreen, hiking boots, and allergy medication.",
          "I already packed the hiking boots, so remove those from the list."],
         [("What do I still need to pack?",
           "You still need to pack your passport, sunscreen, and allergy medication.",
           ["hiking boots"]),
          ("Do I need my passport?",
           "Yes, your passport is still on the packing list.",
           []),
          ("Do I need sunscreen?",
           "Yes, sunscreen is still on your packing list.",
           [])]),
    ]
    for grp, setups, probes in minimal:
        for idx, s in enumerate(setups):
            checks = ["belief_created"] if idx == 0 else ["tension_increased"]
            add(dom, "minimal_change", s, ref="AGM 1985", group=grp,
                setup=True, checks=checks)
        for probe_msg, gold, forbidden in probes:
            add(dom, "minimal_change", probe_msg, gold, ref="AGM 1985",
                group=grp, forbidden=forbidden)
    # extra pair for count
    add(dom, "minimal_change", "My schedule: Monday-gym, Wednesday-dentist, Friday-dinner with Sam.",
        ref="AGM 1985", group="sc_min_4", setup=True, checks=["belief_created"])
    add(dom, "minimal_change", "Dentist moved to Thursday. Everything else stays.",
        ref="AGM 1985", group="sc_min_4", setup=True, checks=["tension_increased"])
    add(dom, "minimal_change", "When is gym day?",
        "Gym day is still Monday.", ref="AGM 1985", group="sc_min_4")
    add(dom, "minimal_change", "When is the dentist now?",
        "The dentist moved to Thursday.", ref="AGM 1985", group="sc_min_4")
    add(dom, "minimal_change", "Is dinner with Sam still Friday?",
        "Yes, dinner with Sam is still on Friday.", ref="AGM 1985", group="sc_min_4")
    # extra minimal_change group
    add(dom, "minimal_change", "Grocery list: apples, bread, chicken, pasta, olive oil.",
        ref="AGM 1985", group="sc_min_5", setup=True, checks=["belief_created"])
    add(dom, "minimal_change", "Replace chicken with tofu. Everything else same.",
        ref="AGM 1985", group="sc_min_5", setup=True, checks=["tension_increased"])
    add(dom, "minimal_change", "What protein am I buying?",
        "You are buying tofu instead of chicken.", ref="AGM 1985", group="sc_min_5",
        forbidden=["buying chicken"])
    add(dom, "minimal_change", "Do I still need bread?",
        "Yes, bread is still on your grocery list.", ref="AGM 1985", group="sc_min_5")
    add(dom, "minimal_change", "What about olive oil?",
        "Yes, olive oil is still on the list. Only chicken changed to tofu.", ref="AGM 1985", group="sc_min_5")

    # --- iterative_revision (20 prompts: 8 setup + 12 probes) ---
    # Multiple consecutive corrections, verify final state is correct
    iterative = [
        ("sc_iter_1",
         ["My favorite color is blue.",
          "Actually, I changed my mind. My favorite color is green.",
          "Wait, I think I like purple best."],
         [("What is my favorite color?", "Your favorite color is purple.",
           ["Your favorite color is blue", "Your favorite color is green"]),
          ("Has my favorite color changed multiple times?",
           "Yes, you changed from blue to green and then to purple.",
           []),
          ("What was my first stated favorite color?",
           "Your first stated favorite color was blue, before changing to green and then purple.",
           [])]),
        ("sc_iter_2",
         ["The party is at 7pm.",
          "Changed to 8pm.",
          "Final answer: 7:30pm."],
         [("What time is the party?", "The party is at 7:30pm.",
           ["The party is at 7pm", "The party is at 8pm"]),
          ("How many times did the party time change?",
           "The party time changed twice: from 7pm to 8pm, then from 8pm to 7:30pm.",
           []),
          ("What was the second proposed time?",
           "The second proposed time was 8pm, before being finalized at 7:30pm.",
           [])]),
        ("sc_iter_3",
         ["We're having steak for dinner.",
          "Change of plans, chicken instead.",
          "Never mind, let's do pasta."],
         [("What's for dinner?", "Dinner is pasta.",
           ["steak", "chicken"]),
          ("Has the dinner plan changed?",
           "Yes, it changed from steak to chicken and finally to pasta.",
           [])]),
    ]
    for grp, setups, probes in iterative:
        for idx, s in enumerate(setups):
            checks = ["belief_created"] if idx == 0 else ["tension_increased"]
            add(dom, "iterative_revision", s, ref="AGM 1985", group=grp,
                setup=True, checks=checks)
        for probe_msg, gold, forbidden in probes:
            add(dom, "iterative_revision", probe_msg, gold, ref="AGM 1985",
                group=grp, forbidden=forbidden)
    # extra sequence for count
    add(dom, "iterative_revision", "My go-to coffee order is a latte.",
        ref="AGM 1985", group="sc_iter_4", setup=True, checks=["belief_created"])
    add(dom, "iterative_revision", "Switched to cappuccino.",
        ref="AGM 1985", group="sc_iter_4", setup=True, checks=["tension_increased"])
    add(dom, "iterative_revision", "Actually, flat white is my new favorite.",
        ref="AGM 1985", group="sc_iter_4", setup=True, checks=["tension_increased"])

    # --- long_horizon_decay (40 prompts) ---
    # horizon="long_horizon" — harness calls POST /bel/iterate between these
    # Tests decay, dormancy, and reawakening mechanics
    decay_groups = [
        # Group 1: high-salience vs low-salience decay
        ("sc_decay_1",
         [("My most important meeting this year is the board review on June 15th.",
           True, ["belief_created"]),
          ("I heard there might be a new coffee shop opening downtown.",
           True, ["belief_created"]),
          # probes after simulated decay
          ("When is my most important meeting?",
           "Your most important meeting, the board review, is on June 15th.",
           False, ["decay_monotonic"]),
          ("What did I say about a coffee shop?",
           "You mentioned there might be a new coffee shop opening downtown.",
           False, ["dormancy_threshold"]),
         ]),
        # Group 2: reinforced vs unreinforced
        ("sc_decay_2",
         [("My daughter's school starts at 8am.",
           True, ["belief_created"]),
          ("My daughter's school starts at 8am. I need to remember this!",
           True, ["confidence_increased", "salience_boosted"]),
          ("I think I saw a hawk in the backyard yesterday.",
           True, ["belief_created"]),
          # probes after decay
          ("What time does my daughter's school start?",
           "Your daughter's school starts at 8am.",
           False, []),
          ("What did I see in the backyard?",
           "You mentioned seeing a hawk in the backyard yesterday.",
           False, ["dormancy_threshold"]),
         ]),
        # Group 3: reawakening dormant belief
        ("sc_decay_3",
         [("I learned that the capital of Bhutan is Thimphu.",
           True, ["belief_created"]),
          # after heavy decay, the belief may go dormant
          ("Actually, remind me — what's the capital of Bhutan?",
           "The capital of Bhutan is Thimphu.",
           False, []),
          # after reawakening, verify the belief is alive again
          ("Tell me again about Bhutan's capital.",
           "The capital of Bhutan is Thimphu.",
           False, []),
         ]),
        # Group 4: multiple beliefs, differential decay
        ("sc_decay_4",
         [("Three things to remember: dentist on the 5th, oil change on the 12th, anniversary on the 20th.",
           True, ["belief_created"]),
          ("The dentist on the 5th is really critical, I can't miss it.",
           True, ["confidence_increased"]),
          # after decay
          ("When is my dentist appointment?",
           "Your dentist appointment is on the 5th.",
           False, []),
         ]),
        # Group 5: tension + decay interaction
        ("sc_decay_5",
         [("The team outing is at the lake.",
           True, ["belief_created"]),
          ("Actually, the team outing might be at the park instead.",
           True, ["tension_increased"]),
          # after decay, the tensioned belief
          ("Where is the team outing?",
           "The team outing location is uncertain — it was initially planned for the lake but may have changed to the park.",
           False, []),
         ]),
        # Group 6: long-term persistent belief
        ("sc_decay_6",
         [("My social security number ends in 4982. This is permanent information.",
           True, ["belief_created"]),
          ("My social security number ends in 4982.",
           True, ["confidence_increased", "salience_boosted"]),
          # after many decay cycles, critical info should persist
          ("What did I say my social security number ends in?",
           "You said your social security number ends in 4982.",
           False, []),
         ]),
        # Group 7: ephemeral vs persistent
        ("sc_decay_7",
         [("I had a sandwich for lunch today.",
           True, ["belief_created"]),
          ("My blood type is AB positive. This never changes.",
           True, ["belief_created"]),
          # after decay
          ("What is my blood type?",
           "Your blood type is AB positive.",
           False, []),
          ("What did I have for lunch?",
           "You had a sandwich for lunch.",
           False, ["dormancy_threshold"]),
         ]),
        # Group 8: belief update + decay
        ("sc_decay_8",
         [("The project deadline is December 1st.",
           True, ["belief_created"]),
          ("The deadline was extended to December 15th.",
           True, ["tension_increased"]),
          # after decay, verify updated belief survives
          ("When is the project deadline?",
           "The project deadline is December 15th.",
           False, []),
          ("What was the original deadline?",
           "The original deadline was December 1st before being extended to December 15th.",
           False, []),
         ]),
        # Group 9: contradicted belief + decay
        ("sc_decay_9",
         [("The team meeting is on Tuesdays.",
           True, ["belief_created"]),
          ("Wait, the team meeting moved to Wednesdays.",
           True, ["tension_increased"]),
          # after decay, the corrected belief should persist
          ("When is the team meeting?",
           "The team meeting is on Wednesdays.",
           False, []),
         ]),
        # Group 10: trivial vs critical info decay
        ("sc_decay_10",
         [("My daughter is severely allergic to peanuts. This is life-critical.",
           True, ["belief_created"]),
          ("My daughter's peanut allergy is serious, I must remember.",
           True, ["confidence_increased", "salience_boosted"]),
          ("What is my daughter allergic to?",
           "Your daughter is severely allergic to peanuts.",
           False, []),
         ]),
    ]
    for grp, items in decay_groups:
        for item in items:
            msg = item[0]
            is_setup = item[1]
            if is_setup is True:
                checks = item[2]
                add(dom, "long_horizon_decay", msg, ref="AGM 1985", group=grp,
                    setup=True, checks=checks, horizon="long_horizon")
            else:
                gold = item[1]
                checks = item[2] if len(item) > 2 else []
                add(dom, "long_horizon_decay", msg, gold, ref="AGM 1985",
                    group=grp, checks=checks, horizon="long_horizon")
