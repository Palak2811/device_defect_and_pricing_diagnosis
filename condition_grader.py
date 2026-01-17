class ConditionGrader:
    def __init__(self):

        self.grade_criteria = {
            'A': {
                'description': 'Excellent - Like New',
                'max_defects': 0,
                'min_condition_score': 9.0,
                'max_severity': 0
            },
            'B': {
                'description': 'Good - Minor Cosmetic Issues',
                'max_defects': 2,
                'min_condition_score': 7.0,
                'max_severity': 5
            },
            'C': {
                'description': 'Fair - Moderate Damage',
                'max_defects': 3,
                'min_condition_score': 5.0,
                'max_severity': 7
            },
            'D': {
                'description': 'Poor - Significant Damage',
                'max_defects': 5,
                'min_condition_score': 3.0,
                'max_severity': 9
            },
            'F': {
                'description': 'Unacceptable - Major Damage/Non-functional',
                'max_defects': 999,
                'min_condition_score': 0,
                'max_severity': 10
            }
        }
    
    def calculate_condition_score(self, defects):
       
        if not defects:
            return 10.0
        total_severity = sum(d['severity_score'] for d in defects)
        num_defects = len(defects)
        avg_severity = total_severity / num_defects
        has_critical = any(d['critical'] for d in defects)
        condition_score = 10 - avg_severity
        if num_defects > 1:
            condition_score -= (num_defects - 1) * 0.5
        if has_critical:
            condition_score -= 2.0
        
        return max(0.0, min(10.0, condition_score))
    
    def assign_grade(self, defects, condition_score=None):
        
        if condition_score is None:
            condition_score = self.calculate_condition_score(defects)
        
        num_defects = len(defects)
        max_severity = max([d['severity_score'] for d in defects], default=0)
        has_critical = any(d.get('critical', False) for d in defects)
        
        if num_defects == 0:
            grade = 'A'
        elif has_critical or max_severity >= 9:
            if num_defects >= 2:
                grade = 'F'
            else:
                grade = 'D'
        elif condition_score >= 9:
            grade = 'A'
        elif condition_score >= 7:
            grade = 'B'
        elif condition_score >= 5:
            grade = 'C'
        elif condition_score >= 3:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'grade': grade,
            'description': self.grade_criteria[grade]['description'],
            'condition_score': round(condition_score, 2),
            'breakdown': {
                'num_defects': num_defects,
                'max_severity': max_severity,
                'has_critical': has_critical,
                'defect_categories': list(set(d['category'] for d in defects))
            }
        }
    
    def get_grade_info(self, grade):
        return self.grade_criteria.get(grade, {})


if __name__ == "__main__":
    grader = ConditionGrader()
    
    