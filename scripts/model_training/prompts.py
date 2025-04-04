FILM_EDITING_PROMPT = """
### Context
You are analyzing film editing patterns. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
1. Determine if the **current shot** matches one or more of the following editing patterns.
2. If no pattern applies, return an empty list.
3. Output the result as a JSON object with the key `"patterns"`.

### Editing Patterns
1. **alternating-shot**:
   - Alternates between two distinct actors, groups, or objects across multiple shots.
   - Each actor/group/object appears at least twice.

2. **cut-away**:
   - Temporarily shifts focus from the main subject (e.g., an actor) to a secondary element (e.g., a group or object).
   - Then returns to the original subject.

3. **cut-in**:
   - Transitions from a wider shot to a closer detail of the same scene or subject.
   - Then returns to the original framing or context.

4. **intensify**:
   - Gradually zooms in (or moves closer) on a face or object over a sequence of shots.

5. **shot-reverse-shot**:
   - Alternates between two characters, typically in dialogue.
   - Focuses on close or medium shots of each character as they speak or react.

### Output Format
Return **only** a JSON object with the key `"patterns"`. Example outputs:
{
  "patterns": ["alternating-shot", "cut-away"]
}
or
{
  "patterns": []
}
"""

FILM_EDITING_DEFS_PROMPT = """
### Context
You are analyzing film editing patterns. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
1. Determine if the **current shot** matches one or more of the following editing patterns.
2. If no pattern applies, return an empty list.
3. Output the result as a JSON object with the key `"patterns"`.

Below are the patterns with formal definitions. Each bullet describes constraints or relations among shots (s1, s2, s3, ...). For example, “actor-relation: same (s1, s3)” means the same actor (or set of actors) appears in shot 1 and shot 3.

### Editing Patterns and Their Definitions:

**alternating-shot**
- Length: ≥ 4
- Actor-or-object-relation: same (s1, s3), same (s2, s4)
- Place-relation: same (s1, s3), same (s2, s4)
- Visual-similarity: low (s1, s2)

**cut-away**
- Length: ≥ 3
- Actor-relation: same (s1, s3)
- Must satisfy one of the following conditions:
  1. Actor-count: ≥ 2 (s2)
     Actor-relation: different (s1, s2)
     Visual-similarity: low (s1, s2)
     Visual-similarity: high (s1, s3)
  2. Actor-count: 0 (s2)
     Place-relation: same (s1, s2)
     Size-relation: closer (s1, s2)
  3. Actor-count: 0 (s2)
     Place-relation: different (s1, s2)

**cut-in**
- Length: = 3
- Actor-relation: same (s1, s3)
- Size-relation: closer (s1, s2)
- Object-relation: same (s1, s2)

**intensify**
- Length: ≥ 2
- Actor-or-object-relation: same (all)
- Size-relation: closer (all)

**shot-reverse-shot**
- Length: ≥ 3
- Actor-count: 1 (all)
- Actor-relation: different (s1, s2)
- Actor-relation: same (s1, s3)
- Gaze: non-direct (all)

### Output Format
Return **only** a JSON object with the key `"patterns"`. Example outputs:
{
  "patterns": ["alternating-shot", "cut-away"]
}
or
{
  "patterns": []
}
"""

ALTERNATING_SHOT = """
### Context
You are analyzing film editing patterns. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
Determine whether the **current shot** is part of an alternating-shots pattern. Output a boolean value (1 or 0).

### Pattern Definition
A sequence of shots where:
- Length: At least 4 shots.
- Alternates between two distinct actors or objects.
- Each actor or object appears at least twice in alternating positions.
- Visual similarity is low between alternating shots (e.g., shot 1 and shot 2).

### Output Format
Return **only** a boolean value:
1 (match)
0 (no match)

### Example Outputs
1
0
"""

CUT_AWAY = """
### Context
You are analyzing film editing patterns. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
Determine whether the **current shot** is part of a cut-away pattern. Output a boolean value (1 or 0).

### Pattern Definition
A sequence of shots where:
- Length: Typically 3 shots, but may vary slightly.
- Shifts focus from the main actor or subject in shot 1 to a secondary element (e.g., a group, object, or detail) in shot 2.
- Returns to the original actor or subject in shot 3.
- EITHER:
    - The secondary element in shot 2 is visually similar to shot 1 but closer in framing (e.g., a detail of the same scene).
  OR:
    - The secondary element in shot 2 is visually distinct, showing a different location, context, or focus.
- Visual similarity is low between shot 1 and shot 2, but high between shot 1 and shot 3.
- Shot 2 often contains multiple faces, focuses on an object, or depicts a non-actor element.

### Output Format
Return **only** a boolean value:
1 (match)
0 (no match)

### Example Outputs
1
0
"""

CUT_IN = """
### Context
You are analyzing film editing patterns. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
Determine whether the **current shot** is part of a cut-in pattern. Output a boolean value (1 or 0).

### Pattern Definition
A sequence of shots where:
- Length: Typically 3 shots, but may vary slightly.
- Begins with a wider shot (shot 1), transitions to a closer view or detail of the same scene or object in shot 2, and returns to the original framing in shot 3.
- The object or scene remains consistent across all shots.
- Shot 2 is visually closer than shot 1.

### Output Format
Return **only** a boolean value:
1 (match)
0 (no match)

### Example Outputs
1
0
"""

INTENSIFY = """
### Context
You are analyzing film editing patterns. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
Determine whether the **current shot** is part of an intensify pattern. Output a boolean value (1 or 0).

### Pattern Definition
A sequence of shots where:
- Length: At least 2 shots, but often extends to several shots.
- Focuses on the same actor or object across all shots.
- Gradually zooms in or moves closer to the actor or object over the sequence.
- Visual framing becomes progressively tighter.

### Output Format
Return **only** a boolean value:
1 (match)
0 (no match)

### Example Outputs
1
0
"""

SHOT_REVERSE_SHOT = """
### Context
You are analyzing film editing patterns. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
Determine whether the **current shot** is part of a shot-reverse-shot pattern. Output a boolean value (1 or 0).

### Pattern Definition
A sequence of shots where:
- Length: At least 3 shots, but often extends to depict ongoing dialogue.
- Alternates between two characters, typically in conversation.
- Each shot focuses on one character, with low visual similarity between alternating shots.
- Characters’ gazes are non-direct (e.g., looking off-screen toward the other character).

### Output Format
Return **only** a boolean value:
1 (match)
0 (no match)

### Example Outputs
1
0
"""

STRATEGIES_PROMPT = """
### Context
You are analyzing media patterns in visual and verbal content. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
1. Determine if the **current shot** matches one or more of the following patterns.
2. If no pattern applies, return an empty list.
3. Output the result as a JSON object with the key `"patterns"`.

### Media Patterns
1. **fragmentation**:
   - Splits visual and/or verbal information to highlight specific story elements or multiple perspectives.
   - Alternates shots of a speaker with shots that do not show the speaker, creating breaks in visual continuity.

2. **fragmentation_splitscreen**:
   - Splits visual and/or verbal information to highlight specific story elements or multiple perspectives.
   - Uses a split-screen format to show simultaneous perspectives or actions.

3. **individualization_of_elite**:
   - Focuses on a single elite individual—someone with power or high public standing—rather than referencing a group or institution (e.g., “the government”).
   - Shows this elite person in a studio context where they are identified by name, followed by a shot of them in a different (non-studio) setting. This emphasizes their personal agency or responsibility.

4. **individualization_of_reporter**:
   - Centers on a reporter in a situation outside the news studio.
   - The reporter references themselves (e.g., “My opinion is…”), moving from a neutral information source to a more personal and possibly emotional figure.

5. **individualization_of_layperson**:
   - Focuses on someone from the general public—an everyday individual. The viewer sees and hears this person, giving them presence and voice within the narrative.
   - Typically shows the layperson in a non-studio context (e.g., at home or work), emphasizing their personal experience and distinguishing them from professional media figures.

6. **emotionalization**:
   - Heightens the emotional tone of a topic.
   - Displays the same emotion (e.g., sadness, shock, excitement) across multiple shots to underscore its significance in the narrative.

### Output Format
Return **only** a JSON object with the key `"patterns"`. Example outputs:
{
  "patterns": ["fragmentation", "individualization_of_layperson"]
}
or
{
  "patterns": []
}
"""

STRATEGIES_DEFS_PROMPT = """
### Context
You are analyzing media patterns in visual and verbal content. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
1. Determine if the **current shot** matches one or more of the following patterns.
2. If no pattern applies, return an empty list.
3. Output the result as a JSON object with the key `"patterns"`.

Below are the patterns with formal definitions. Each bullet describes constraints or relations among shots (s1, s2, s3, ...). For example, “actor-relation: same (s1, s3)” means the same actor (or set of actors) appears in shot 1 and shot 3.

### Media Patterns and Their Definitions:

**fragmentation**
- length: >= 4
- speaker-relation: same (all)
- shot1 + shot3: actor-relation = same, place-relation = same, talkspace = on-screen
- shot2 + shot4: object-relation = same, place-relation = same, talkspace = off-screen
- shot1 + shot2: visual-similarity = low

**fragmentation_splitscreen**
- length: = 1
- actor.role: != reporter
- talkspace: on-screen
- splitscreen: true

**individualization_of_elite**
- length: >= 2
- shot1: place = studio, talkspace = on-screen, spoken_text >= 1 named entity (person)
- shot2: place != studio, talkspace = off-screen, actor.role != {reporter, anchor}
- actor-relation: same (all)

**individualization_of_layperson**
- length: >= 2
- place: != studio
- last_shot: actor.role != {reporter, anchor}, talkspace = on-screen
- actor-relation: at least one actor same (all)

**individualization_of_reporter**
- length: >= 1
- last_speaker_turn: spoken_text >= 1 self-referent-pronoun
- last_shot: talkspace = on-screen
- place != studio
- actor.role = reporter

**emotionalization**
- length: >= 2
EITHER
   face-emotion: not neutral
   face-emotion-relation: same (all)
OR
   sentiment: highly positive/negative


### Output Format
Return **only** a JSON object with the key `"patterns"`. Example outputs:
{
  "patterns": ["fragmentation", "individualization_of_layperson"]
}
or
{
  "patterns": []
}
"""


FRAGMENTATION = """
### Context
You are analyzing media patterns in visual and verbal content. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
Determine whether the **current shot** is part of a fragmentation pattern. Output a boolean value (1 or 0).

### Pattern Definition
A sequence of shots where:
- Length: At least 4 shots.
- The speaker remains the same across all shots.
- Shot 1 and shot 3 show the speaker in the same place.
- Shot 2 and shot 4 show the same object in the same place but differ from shot 1 in visual content.

### Output Format
Return **only** a boolean value:
1 (match)
0 (no match)

### Example Outputs
1
0
"""

FRAGMENTATION_SPLITSCREEN = """
### Context
You are analyzing media patterns in visual and verbal content. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
Determine whether the **current shot** is part of a fragmentation_splitscreen pattern. Output a boolean value (1 or 0).

### Pattern Definition
A single shot where:
- The actor is not a reporter.
- The speaker is on-screen.
- The shot uses a split-screen format to show simultaneous perspectives or actions.

### Output Format
Return **only** a boolean value:
1 (match)
0 (no match)

### Example Outputs
1
0
"""

INDIVIDUALIZATION_OF_ELITE = """" " 
### Context
You are analyzing media patterns in visual and verbal content. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
Determine whether the **current shot** is part of an individualization_of_elite pattern. Output a boolean value (1 or 0).

### Pattern Definition
A sequence of shots where:
- Length: At least 2 shots.
- Shot 1 shows an elite individual in a studio context, identified by name.
- Shot 2 shows the same elite individual in a different setting.
- Both shots focus on the same elite individual.

### Output Format
Return **only** a boolean value:
1 (match)
0 (no match)

### Example Outputs
1
0
"""

INDIVIDUALIZATION_OF_LAYPERSON = """
### Context
You are analyzing media patterns in visual and verbal content. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
Determine whether the **current shot** is part of an individualization_of_layperson pattern. Output a boolean value (1 or 0).

### Pattern Definition
A sequence of shots where:
- Length: At least 2 shots.
- The place is not a studio.
- The last shot does not feature a reporter or anchor and has on-screen talkspace.
- At least one actor remains the same across all shots.

### Output Format
Return **only** a boolean value:
1 (match)
0 (no match)

### Example Outputs
1
0
"""
INDIVIDUALIZATION_OF_REPORTER = """
### Context
You are analyzing media patterns in visual and verbal content. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
Determine whether the **current shot** is part of an individualization_of_reporter pattern. Output a boolean value (1 or 0).

### Pattern Definition
A sequence of shots where:
- Length: At least 1 shot.
- Has a reporter talking on-screen outside a studio setting.
- The reporter references themselves in the spoken text.

### Output Format
Return **only** a boolean value:
1 (match)
0 (no match)

### Example Outputs
1
0
"""

EMOTIONALIZATION = """
### Context
You are analyzing media patterns in visual and verbal content. Inputs include:
- The two previous shots (if available).
- The current shot (to classify).
- The next two shots (if available).
- Dialogue transcripts for each speaker-turn during the shots.

### Task
Determine whether the **current shot** is part of an emotionalization pattern. Output a boolean value (1 or 0).

### Pattern Definition
A sequence of shots where:
- Length: At least 2 shots.
- EITHER:
    - The face emotion is not neutral across all shots.
    - The face emotion remains the same across all shots.
- OR:
    - The sentiment is highly positive or negative.

### Output Format
Return **only** a boolean value:
1 (match)
0 (no match)

### Example Outputs
1
0
"""
