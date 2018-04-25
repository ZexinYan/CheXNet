static_tags_name = ['nodule', 'borderline', 'spinal fusion', 'cardiac shadow', 'interstitial', 'pulmonary congestion',
                    'technical quality of image unsatisfactory', 'bronchiectasis', 'cervical vertebrae',
                    'hypoinflation', 'medical device', 'prominent', 'mass', 'breast implants', 'calcinosis',
                    'aortic aneurysm', 'aorta, thoracic', 'lower lobe', 'scattered', 'left', 'lung, hyperlucent',
                    'pneumoperitoneum', 'enlarged', 'foreign bodies', 'epicardial fat', 'reticular', 'abnormal',
                    'irregular', 'obscured', 'large', 'diaphragm', 'right', 'breast', 'edema',
                    'hyperostosis, diffuse idiopathic skeletal', 'airspace disease', 'stents', 'mild', 'volume loss',
                    'shift', 'sulcus', 'humerus', 'lucency', 'blunted', 'osteophyte', 'blood vessels',
                    'lumbar vertebrae', 'flattened', 'tortuous', 'small', 'healed', 'hypertension, pulmonary',
                    'bone diseases, metabolic', 'trachea', 'atherosclerosis', 'mediastinum', 'coronary vessels', 'lung',
                    'chronic', 'multiple', 'ribs', 'pulmonary disease, chronic obstructive', 'apex', 'hilum',
                    'spondylosis', 'diffuse', 'paratracheal', 'pneumothorax', 'clavicle', 'retrocardiac', 'lymph nodes',
                    'bronchovascular', 'azygos lobe', 'emphysema', 'granulomatous disease',
                    'calcified granuloma', 'normal', 'thoracic vertebrae', 'funnel chest', 'thorax', 'aorta',
                    'adipose tissue', 'anterior', 'arthritis', 'emphysema', 'fractures, bone', 'hernia',
                    'implanted medical device', 'sutures', 'granuloma', 'pleura', 'thickening', 'cysts', 'upper lobe',
                    'middle lobe', 'effusion', 'deformity', 'contrast media', 'atelectasis',
                    'hyperdistention', 'effusion', 'spine', 'mastectomy', 'surgical instruments',
                    'nipple shadow', 'heart', 'streaky', 'blister', 'catheters, indwelling', 'bilateral', 'neck',
                    'cavitation', 'density', 'scoliosis', 'pulmonary artery', 'round', 'opacity',
                    'lung diseases, interstitial', 'sternum', 'heart ventricles', 'lingula', 'aortic valve',
                    'heart failure', 'heart atria', 'sarcoidosis', 'emphysema', 'sclerosis',
                    'costophrenic angle', 'kyphosis', 'hydropneumothorax', 'consolidation', 'dislocations', 'markings',
                    'abdomen', 'tube, inserted', 'no indexing', 'pneumonectomy', 'posterior', 'patchy',
                    'diaphragmatic eventration', 'fibrosis', 'pneumonia', 'cardiomegaly', 'focal', 'cicatrix',
                    'elevated', 'infiltrate', 'moderate', 'degenerative', 'base', 'trachea, carina', 'severe',
                    'bronchi', 'pulmonary alveoli', 'shoulder', 'fibrosis']

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Thickening', 'Hernia']


if __name__ == '__main__':
    for i, each in enumerate(CLASS_NAMES):
        CLASS_NAMES[i] = each.lower()
    print(CLASS_NAMES)
    sets = set()
    for each in static_tags_name:
        if each in CLASS_NAMES:
            sets.add(each)
    print(len(sets))
