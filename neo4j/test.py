
def test_overview():
    # simple case
    pass

def test_cause():
    pass

def test_symptom():
    pass

def test_risk_factor():
    pass

def test_treatment():
    pass

def test_prevention():
    pass

def test_severity():
    pass

def test_diag():
    pass

def test_verify():
    pass

if __name__ == '__main__':
    inferencer = Inferencer()

    request = {
        'symptom': ['phân có máu','sốt','chóng mặt','buồn nôn','đau ngực'],
        'disease': ['trĩ ngoại'],
        'intent' : 'diag'
    }
    answer = inferencer.query(request)
    print(answer)