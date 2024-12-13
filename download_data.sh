mkdir sample_data
mkdir -p output output/HumanoidIm/ output/HumanoidIm/omnigrasp_neurips_grab/ output/HumanoidIm/pulse_x_omnigrasp/
gdown https://drive.google.com/uc?id=1bLp4SNIZROMB7Sxgt0Mh4-4BLOPGV9_U -O  sample_data/ # filtered shapes from AMASS
gdown https://drive.google.com/uc?id=1arpCsue3Knqttj75Nt9Mwo32TKC4TYDx -O  sample_data/ # all shapes from AMASS
gdown https://drive.google.com/uc?id=1fFauJE0W0nJfihUvjViq9OzmFfHo_rq0 -O  sample_data/ # sample standing neutral data.
gdown https://drive.google.com/uc?id=1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc -O  sample_data/ # amass_occlusion_v3

gdown https://drive.google.com/uc?id=1vUb7-j_UQRGMyqC_uY0YIdy6May297K5 -O  sample_data/ # PHC_X standing 
gdown https://drive.google.com/uc?id=1zmiiGn6TyNQp4UISP8Ra-bTExdlGhBbn -O  sample_data/ # Hammer

gdown https://drive.google.com/uc?id=1HdC4Vk44_7NUiZ39xmjb2sRP7OMwNnNq -O  output/HumanoidIm/pulse_x_omnigrasp/ # PHC_X standing 
gdown https://drive.google.com/uc?id=1QbYl11wmFJgvqeAoBvoZ8-R9wVy0833Q -O  output/HumanoidIm/omnigrasp_neurips_grab/ # PHC_X standing 

