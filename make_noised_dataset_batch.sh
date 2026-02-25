#!/bin/zsh

# python make_noised_testset.py --data_dir ./data/test \
#     --user_speech_noise \
#     --user_speech_noise_prob 0.3 \
#     --user_speech_noise_mean_sec 1.0 \
#     --user_speech_snr_db 60.0 \
#     --user_speech_silence \
#     --user_speech_silence_prob 0.3 \
#     --user_speech_silence_mean_sec 1.0 \
#     --agent_speech_noise \
#     --agent_speech_noise_prob 0.3 \
#     --agent_speech_noise_mean_sec 1.0 \
#     --agent_speech_snr_db 60.0 \
#     --agent_speech_silence \
#     --agent_speech_silence_prob 0.3 \
#     --agent_speech_silence_mean_sec 1.0 \
#     --user_motion_drop_prob 0.3 \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0.3 \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type gaussian 

####### User Head Motion Drop
# python make_noised_testset.py --data_dir ./data/test \
#     --user_motion_drop_prob 0. \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0. \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type gaussian 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_motion_drop_prob 0.1 \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0. \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type gaussian 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_motion_drop_prob 0.3 \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0. \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type gaussian 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_motion_drop_prob 0.5 \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0. \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type gaussian 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_motion_drop_prob 0.7 \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0. \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type gaussian 

###### User Head Motion Noise
# python make_noised_testset.py --data_dir ./data/test \
#     --user_motion_drop_prob 0. \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0.1 \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type gaussian 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_motion_drop_prob 0. \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0.3 \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type gaussian 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_motion_drop_prob 0. \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0.5 \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type gaussian 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_motion_drop_prob 0. \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0.7 \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type gaussian 
# ############# User Head Motion Noise Type
# python make_noised_testset.py --data_dir ./data/test \
#     --user_motion_drop_prob 0. \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0.7 \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type laplace 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_motion_drop_prob 0. \
#     --user_motion_drop_mean_sec 1.0 \
#     --user_motion_noise_prob 0.7 \
#     --user_motion_noise_mean_sec 1.0 \
#     --user_motion_noise_type uniform 


# ######## User Speech Silence
# python make_noised_testset.py --data_dir ./data/test \
#     --user_speech_silence \
#     --user_speech_silence_prob 0.1 \
#     --user_speech_silence_mean_sec 1.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_speech_silence \
#     --user_speech_silence_prob 0.3 \
#     --user_speech_silence_mean_sec 1.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_speech_silence \
#     --user_speech_silence_prob 0.5 \
#     --user_speech_silence_mean_sec 1.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_speech_silence \
#     --user_speech_silence_prob 0.7 \
#     --user_speech_silence_mean_sec 1.0 

# ######## User Speech Noise
# python make_noised_testset.py --data_dir ./data/test \
#     --user_speech_noise \
#     --user_speech_noise_prob 0.1 \
#     --user_speech_noise_mean_sec 1.0 \
#     --user_speech_snr_db 60.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_speech_noise \
#     --user_speech_noise_prob 0.3 \
#     --user_speech_noise_mean_sec 1.0 \
#     --user_speech_snr_db 60.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_speech_noise \
#     --user_speech_noise_prob 0.5 \
#     --user_speech_noise_mean_sec 1.0 \
#     --user_speech_snr_db 60.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_speech_noise \
#     --user_speech_noise_prob 0.7 \
#     --user_speech_noise_mean_sec 1.0 \
#     --user_speech_snr_db 60.0 



# ######## Agent Speech Silence
# python make_noised_testset.py --data_dir ./data/test \
#     --agent_speech_silence \
#     --agent_speech_silence_prob 0.1 \
#     --agent_speech_silence_mean_sec 1.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --agent_speech_silence \
#     --agent_speech_silence_prob 0.3 \
#     --agent_speech_silence_mean_sec 1.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --agent_speech_silence \
#     --agent_speech_silence_prob 0.5 \
#     --agent_speech_silence_mean_sec 1.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --agent_speech_silence \
#     --agent_speech_silence_prob 0.7 \
#     --agent_speech_silence_mean_sec 1.0 

# ######## Agent Speech Noise
# python make_noised_testset.py --data_dir ./data/test \
#     --agent_speech_noise \
#     --agent_speech_noise_prob 0.1 \
#     --agent_speech_noise_mean_sec 1.0 \
#     --agent_speech_snr_db 60.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --agent_speech_noise \
#     --agent_speech_noise_prob 0.3 \
#     --agent_speech_noise_mean_sec 1.0 \
#     --agent_speech_snr_db 60.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --agent_speech_noise \
#     --agent_speech_noise_prob 0.5 \
#     --agent_speech_noise_mean_sec 1.0 \
#     --agent_speech_snr_db 60.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --agent_speech_noise \
#     --agent_speech_noise_prob 0.7 \
#     --agent_speech_noise_mean_sec 1.0 \
#     --agent_speech_snr_db 60.0 


python make_noised_testset.py --data_dir ./data/test \
    --agent_speech_silence \
    --agent_speech_silence_prob 1 \
    --agent_speech_silence_mean_sec 4.0 
# python make_noised_testset.py --data_dir ./data/test \
#     --user_speech_noise \
#     --user_speech_noise_prob 1 \
#     --user_speech_noise_mean_sec 4.0 \
#     --user_speech_snr_db 60.0 