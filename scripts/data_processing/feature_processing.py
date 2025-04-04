import numpy as np

ALL_AUDIO_LABELS = {}
def process_feature(shot_dict: dict, feature_name: str, feature_data:dict):
   
    if feature_name in FEATURE_PROCESSORS:
        FEATURE_PROCESSORS[feature_name](shot_dict, feature_name, feature_data)
    else:
        # No known processor => skip or handle gracefully
        print(f"No processor for {feature_name}. Skipping.")
        pass


def process_whisperx_ner(shot_dict, feature_name, feature_data):
    """
    Process the whisperx_ner data and update shot_dict.
    """
    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    # The data structure you'd have in ner_data could vary;
    # adapt as needed:
    entities_vector =  [0, 0, 0, 0, 0, 0]
    feature_data = feature_data['output_data']['speakerturn_wise']
    for entity_data in feature_data:
        if entity_data['start'] < shot_end and entity_data['end'] > shot_start:
            if entity_data['vector'] is not None: # if there is no vector we skip it
                entities_vector = np.add(entities_vector, entity_data['vector'])
    shot_dict[feature_name] = list(entities_vector)
                
def process_whisperx_pos(shot_dict, feature_name, feature_data):
    """
    Process the whisperx_pos data and update shot_dict.
    """
    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    # The data structure you'd have in ner_data could vary;
    # adapt as needed:
    entities_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    feature_data = feature_data['output_data']['speakerturn_wise']
    for pos_turn in feature_data:
        if pos_turn['start'] <= shot_end and pos_turn['end'] >= shot_start:
            if pos_turn['vector'] is not None:
                entities_vector = np.add(entities_vector, pos_turn['vector'])
    shot_dict[feature_name] = list(entities_vector)

def process_whisperx_sentiment(shot_dict, feature_name, sentiment_data):
    """
    Example: if you have sentiment results (pos/neg/neu).
    """
    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    sentiment_data = sentiment_data['output_data']['model_news']['speakerturn_wise']
    predictions = []
    for sentiment_data_segment in sentiment_data:
        if sentiment_data_segment['start'] <= shot_end and sentiment_data_segment['end'] >= shot_start:
            # I want to do it absed on the probability, it needs to be higher than a threshold
            # for now just use 0.5
            prediction_prob = np.array(sentiment_data_segment['prob'])
            if np.max(prediction_prob) and np.max(prediction_prob) > 0.5:
                # the index of the max value is the prediction
                prediction = int(np.argmax(prediction_prob))
            else:
                prediction = int(1) # use neutral as default for now
            # scale predictio n to 0-1
            prediction = prediction / 2
            predictions.append(prediction)
    if predictions:
        shot_dict[feature_name].append(float(np.mean(predictions)))
    else:
        shot_dict[feature_name].append(0.5)


def process_llm_evaluative(shot_dict, feature_name, llm_data):
    """
    I calculate a score based on the evaluative label and confidence.
    Score is between 0 and 1.
    0 is not evaluative (high confidence), 1 is evaluative (high confidence).
    around 0.5 is if the confidence is low.
    for multiple speakers we average the scores.
    """

    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    confidence_dict = {
        'None': 0,
        'low': 0.1,
        'moderate': 0.5,
        'high': 1
    }
    shot_values = []
    llm_data = llm_data['output_data']
    for llm_data_segment in llm_data:
        if llm_data_segment['start'] <= shot_end and llm_data_segment['end'] >= shot_start:
            evaluative = 0.5 if llm_data_segment['label'] == 'evaluative' else -0.5
            confidence = llm_data_segment['confidence']
            if confidence:
                confidence = confidence_dict[confidence]
            else:
                confidence = 0
            evaluative_score = (evaluative * confidence)
            shot_values.append(evaluative_score)
    if shot_values:
        evaluative_score = np.mean(shot_values) + 0.5
        if evaluative_score >1 or evaluative_score < 0:
            print("Error! Evaluative score out of range")
            print(evaluative_score, shot_values, shot_dict['shot_id'])
        else:
            shot_dict[feature_name].append(float(evaluative_score))
    else:
        shot_dict[feature_name].append(0.5)
            
def process_shot_density(shot_dict, feature_name, shot_density_data):
    """
    Process the shot_density data and update shot_dict.
    """
    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    shot_density_data = shot_density_data['output_data']
    density_values = shot_density_data['y']
    timestamps = shot_density_data['time']
    shot_density, occurence_counter = 0, 0
    for timestamp, shot_density_segment in zip(timestamps, density_values):
        if timestamp <= shot_end and timestamp >= shot_start:
            shot_density += shot_density_segment
            occurence_counter += 1
    if occurence_counter > 0:
        shot_density = shot_density / occurence_counter
    else:
        shot_density = 0.5 # default value
    shot_dict[feature_name].append(shot_density)


def process_face_emotions_deepface(shot_dict, feature_name, face_emotion_data):
    """
    Process the face_emotion_deepface data and update shot_dict.
    """

    # If there is no face in the shot we simply have no frame data for the shot in the featuer 
    
    one_hot_encoding = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    # Basically I want all emotions in the encoding that occur strongly at least in one frame
    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    face_emotion_data = face_emotion_data['y']
    shot_dict[feature_name] = np.zeros(7)
    emotion_counter = np.zeros(7)
    for frame_data in face_emotion_data:
        #problem here we have only one time instead of start and end
        if frame_data['time'] <= shot_end and frame_data['time'] >= shot_start:
            frame_emotions = frame_data['emotions']
            # frame_emotions is a dict with emotions and their probabilities
            # I want to add an emotion to the list if it is strong enough
            for emotion in frame_emotions:
                if frame_emotions[emotion] > 0.5:
                    shot_dict[feature_name][one_hot_encoding[emotion]] += 1
                    emotion_counter[one_hot_encoding[emotion]] += 1
    # normalize the vector
    if np.sum(emotion_counter) > 0:
        shot_dict[feature_name] = list(shot_dict[feature_name] / np.sum(emotion_counter))
    else:
        shot_dict[feature_name] = list(shot_dict[feature_name]) # if there are no emotions or no features at all its all zeros

def process_videoshot_scalemovement(shot_dict, feature_name, scale_movement_data):
    """
    Process the scale_movement data and update shot_dict.
    """
    #ECS (Extreme close-up shot), CS (close-up shot), MS (medium shot), FS (full shot), LS (long shot),could be mapped to 0-4 or even 0-1
    shot_size_dict = {
        'ECS': 0,
        'CS': 0.25,
        'MS': 0.5,
        'FS': 0.75,
        'LS': 1
    }

    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    scale_movement_data = scale_movement_data['output_data']
    shot_counter = 0
    for scale_movement_segment in scale_movement_data:
        if scale_movement_segment['shot']['start'] <= shot_end and scale_movement_segment['shot']['end'] >= shot_start:
            shot_counter += 1
            if shot_counter > 1:
                print("More than one segment for one shot")
                print(shot_dict['shot_id', feature_name])
                helper(shot_dict, feature_name, scale_movement_data)
            shot_size = scale_movement_segment['prediction'][0]
            shot_score = shot_size_dict[shot_size]
            shot_dict[feature_name].append(shot_size_dict[shot_size])

def helper(shot_dict, feature_name, audio_clf_data):
    """
    Process the audio_clf data and update shot_dict.
    """
    # All the labels that occur with a probability higher than 0.5 and how often they occur with that probability
    #{'Speech': 181
    print("Helper function")


def process_whisperxspeaker_audioclf(shot_dict, feature_name, audio_clf_data):
    """
    Process the audio_clf data and update shot_dict.

    for label, prob in zip(labels, probs):
                if prob < 0.5:
                    continue
                if label not in ALL_AUDIO_LABELS:
                    ALL_AUDIO_LABELS[label] = 1
                else:
                    ALL_AUDIO_LABELS[label] += 1"""
    # All the labels that occur with a probability higher than 0.5 and how often they occur with that probability
    #{'Speech': 181451, 'Male speech, man speaking': 5307,  'Narration, monologue': 5702, 'Music': 13058, 'Speech synthesizer': 1229, 'Artillery fire': 460, 
    # 'Train': 145, 'Bleat': 145, 'Sheep': 145, 'Whip': 145, 'Gasp': 294, 'Female speech, woman speaking': 2845, 'Rapping': 146,  'Silence': 1337, 'Bird': 316, 
    # 'Telephone bell ringing': 162,  'Bow-wow': 486, 'Run': 263,  'Vehicle': 346, 'Police car (siren)': 147, 'Siren': 147, 'Aircraft': 126, 'Fixed-wing aircraft, airplane': 126,
    #  'Horse': 141, 'Pigeon, dove': 154, 'Emergency vehicle': 51}
    # I want to map some of them to a smaller set of labels

    label_mapping = {
        'Speech': 'Speech',
        'Narration, monologue': 'Narration',
        'Male speech, man speaking': 'Speech',
        'Female speech, woman speaking': 'Speech',
        'Speech synthesizer': 'Speech',
        'Aircraft': 'Vehicle',
        'Fixed-wing aircraft, airplane': 'Vehicle',
        'Police car (siren)': 'Siren',
        'Siren': 'Siren',
        'Vehicle': 'Vehicle',
        'Train': 'Vehicle',
        'Emergency vehicle': 'Siren',
        'Music': 'Music',
        'Bird': 'Animal',
        'Pigeon, dove': 'Animal',
        'Sheep': 'Animal',
        'Horse': 'Animal',
        'Bow-wow': 'Other_Sound',
        'Bleat': 'Other_Sound',
        'Whip': 'Other_Sound',
        'Gasp': 'Other_Sound',
        'Run': 'Other_Sound',
        'Rapping': 'Music',
        'Silence': 'Silence',
        'Telephone bell ringing': 'Other_Sound',
        'Artillery fire': 'Artillery_fire'
    }

    label_to_one_hot = {
        'Speech': 0,
        'Narration': 1,
        'Music': 2,
        'Animal': 3,
        'Vehicle': 4,
        'Siren': 5,
        'Other_Sound': 6,
        'Silence': 7,
        'Artillery_fire': 8
    }

    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    audio_clf_data = audio_clf_data['output_data']
    output_labels = np.zeros(len(label_to_one_hot))
    for audio_clf_segment in audio_clf_data:
        if audio_clf_segment['start'] <= shot_end and audio_clf_segment['end'] >= shot_start:
            labels = audio_clf_segment['top3_label']
            probs = audio_clf_segment['top3_label_prob']
            if not labels:
                continue
            for label, prob in zip(labels, probs):
            
                if prob > 0.5:   
                    label = label_mapping[label]
                    label_id = label_to_one_hot[label]
                    output_labels[label_id] = 1 

    shot_dict[feature_name] = list(output_labels)

def process_whisperxspeaker_segmentClf(shot_dict, feature_name, segment_clf_data):
    # since there can be multiple speakers in one shot, we use a list that gets set to 1 if any speaker of
    # that gender is present in the shot with a probability higher than 0.7

    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    segment_clf_data = segment_clf_data['output_data']
    speakers_gender = [0,0]
    for segment_clf_segment in segment_clf_data:
        if segment_clf_segment['start'] <= shot_end and segment_clf_segment['end'] >= shot_start:
            gender = segment_clf_segment['gender_pred']
            gender_prob = segment_clf_segment['gender_prob']
            if gender_prob and gender_prob > 0.7:
                gender = 1 if gender == 'Female' else 0 # Assuming that gender is considered binary here
                speakers_gender[gender] = 1
    shot_dict[feature_name] = speakers_gender

def process_speaker_turn(shot_dict, feature_name, speakerturn_data):
    # i think gulall used the clip data as basis for this speaker turn data so i will get it from here
    # i want to get whether the speaker is active (0,1) visible on screen
    """
                "anchor",
                "reporter",
                "police-frwr",
                "doctor-nurse",
                "layperson-x-covid",
                "layperson-4-covid",
                "layperson",
                "expert-medicine",
                "expert-other",
                "politician",
                "politician-ampel",
                "politician-right",
                "politician-other",
                "politician-n-de",
                "celeb",
                "talking-head",
                "interview",
                "speech",
                "commenting",
                "voiceover",
            """
    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    speakerturn_data = speakerturn_data['output_data']
    active_acc, speaker_counter, unique_speaker_counter = 0, 0, 0
    heared_speakers = []
    roles_mapping = {
       "anchor": "anchor",
        "reporter": "reporter",
        "police-frwr": "expert",
        "doctor-nurse": "expert",
        "layperson-x-covid": "layperson",
        "layperson-4-covid": "layperson",
        "layperson": "layperson",
        "expert-medicine": "expert",
        "expert-other": "expert",
        "politician": "elite",
        "politician-ampel": "elite",
        "politician-right": "elite",
        "politician-other": "elite",
        "politician-n-de": "elite",
        "celeb": "elite",
    } # the mapping leaves 5 roles anchor, reporter, expert, layperson, elite

    roles_counter = {
        "anchor": 0,
        "reporter": 0,
        "expert": 0,
        "layperson": 0,
        "elite": 0
    }

    situations_counter = {
        "interview": 0,
        "talkin-head": 0,
        "speech": 0,
        "commenting": 0,
        "voiceover": 0
    }
         
    for speakerturn_segment in speakerturn_data:
        if speakerturn_segment['start'] <= shot_end and speakerturn_segment['end'] >= shot_start:
            active_acc += speakerturn_segment['active']
            speaker_counter += 1
            if speakerturn_segment['speaker'] not in heared_speakers:
                heared_speakers.append(speakerturn_segment['speaker'])
                unique_speaker_counter += 1
            role = speakerturn_segment['role_l0']
            if role in roles_mapping:
                roles_counter[roles_mapping[role]] += 1
            situation = speakerturn_segment['situation']
            if situation in situations_counter:
                situations_counter[situation] += 1
    roles = list(roles_counter.values())
    situations = list(situations_counter.values())
    #norm the values to 0-1
    if speaker_counter > 0:
        active_acc = active_acc / speaker_counter
        roles = [role / speaker_counter for role in roles]
        situations = [situation / speaker_counter for situation in situations]
    combined_results = [active_acc, unique_speaker_counter] + roles + situations
    shot_dict[feature_name] = combined_results


def process_shot_similarity(shot_dict, feature_name, shot_similarity_data):
    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    shot_similarity_data = shot_similarity_data['output_data']
    shot_similarity = [0,0,0,0] # always two previous and two following shots
    for shot_similarity_segment in shot_similarity_data:
        if shot_similarity_segment['shot']['start'] <= shot_end and shot_similarity_segment['shot']['end'] >= shot_start:
            # handle edge cases for first and last shots with default = 0.5
            # could make this faster e.g. with dict.get but it works so i might change it only if i have time
            try:
                shot_similarity[0] = shot_similarity_segment['prev_2'][0]
            except KeyError:
                shot_similarity[0] = 0.5
            try:
                shot_similarity[1] = shot_similarity_segment['prev_1'][0]
            except KeyError:
                shot_similarity[1] = 0.5
            try:
                shot_similarity[2] = shot_similarity_segment['next_1'][0]
            except KeyError:
                shot_similarity[2] = 0.5
            try:
                shot_similarity[3] = shot_similarity_segment['next_2'][0]
            except KeyError:
                shot_similarity[3] = 0.5
    shot_dict[feature_name] = shot_similarity

def process_action_similarity(shot_dict, feature_name, shot_similarity_data):
    """
    Process the action similarity data and update shot_dict.
    originally was only supposed to be used for the action similarity but 
    audio similarity has the same format so is also processed this way
    """
    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    shot_similarity_data = shot_similarity_data['output_data']
    shot_similarity = [0,0,0,0] # always two previous and two following shots
    for shot_similarity_segment in shot_similarity_data:
        if shot_similarity_segment['shot']['start'] <= shot_end and shot_similarity_segment['shot']['end'] >= shot_start:
            # handle edge cases for first and last shots with default = 0.5
            # could make this faster e.g. with dict.get but it works so i might change it only if i have time
            try:
                shot_similarity[0] = float(shot_similarity_segment['prev_2'])
            except KeyError:
                shot_similarity[0] = 0.5
            try:
                shot_similarity[1] = float(shot_similarity_segment['prev_1'])
            except KeyError:
                shot_similarity[1] = 0.5
            try:
                shot_similarity[2] = float(shot_similarity_segment['next_1'])
            except KeyError:
                shot_similarity[2] = 0.5
            try:
                shot_similarity[3] = float(shot_similarity_segment['next_2'])
            except KeyError:
                shot_similarity[3] = 0.5
    shot_dict[feature_name] = shot_similarity

def process_face_analysis(shot_dict, feature_name, face_data):
    """
    Process the face analysis data and update shot_dict.
    Currently I only calculate the face size 
    """
    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']
    face_data = face_data['faces']

    acc_face_size, face_counter, unique_face_counter = 0, 0, 0
    seen_faces = []
    regions = [] # upper_left = [1,1], upper_right = [1,0], lower_left = [0,1], lower_right = [0,0]
    #based on 4 region split in wus paper
    for face in face_data:
        if face['time'] <= shot_end and face['time'] >= shot_start:
            face_id = face['face_id']
            if face_id not in seen_faces:
                seen_faces.append(face_id)
                unique_face_counter += 1
            x, y = face['bbox']['x'], face['bbox']['y']
            upper = 1 if y < 0.5 else 0
            left = 1 if x < 0.5 else 0
            region = [upper, left]
            regions.append(region)
            width, height = face['bbox']['w'], face['bbox']['h']
            face_size = width * height
            acc_face_size += face_size
            face_counter += 1

    if face_counter > 0:
        avg_face_size = acc_face_size / face_counter
        avg_region = list(np.mean(regions, axis=0))
        unique_face_counter = unique_face_counter
        # might do the scaling in the feature vectors file when i know the max values
    else:
        avg_face_size = 0 # no face, no size
        avg_region = [0.5, 0.5] # no face, no region but since 0,0 means lower right we use 0.5,0.5 as default

    combined_results = [avg_face_size, unique_face_counter] + avg_region
    shot_dict[feature_name] = combined_results


FEATURE_PROCESSORS = {
    "whisperx_ner.pkl":        process_whisperx_ner,
    "whisperx_pos.pkl":        process_whisperx_pos,
    "whisperx_sentiment.pkl":  process_whisperx_sentiment,
    "llm_evaluative.pkl":    process_llm_evaluative,
    "face_emotions_deepface.pkl": process_face_emotions_deepface,
    "videoshot_scalemovement.pkl": process_videoshot_scalemovement,
    "whisperxspeaker_audioclf.pkl": process_whisperxspeaker_audioclf,
    "whisperxspeaker_segmentClf.pkl": process_whisperxspeaker_segmentClf,
    "speaker_turns_meta.pkl": process_speaker_turn,
    "shot_density.pkl": process_shot_density,
    "siglip_shot_similarity.pkl": process_shot_similarity,
    "convnextv2_shot_similarity.pkl": process_shot_similarity,
    "places_shot_similarity.pkl": process_shot_similarity,
    "kinetics-vmae_action_shot_similarity.pkl": process_action_similarity,
    "ssv2-vmae_action_shot_similarity.pkl": process_action_similarity,
    "kinetics-xclip_action_shot_similarity.pkl": process_action_similarity,
    "wav2vec2_audio_shot_similarity.pkl": process_action_similarity,
    "beats_audio_shot_similarity.pkl": process_action_similarity,
    "whisper_audio_shot_similarity.pkl": process_action_similarity,
    "face_analysis.pkl": process_face_analysis
    # etc...
}
