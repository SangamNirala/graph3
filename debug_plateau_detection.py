#!/usr/bin/env python3
"""
Debug plateau detection for square waves
"""

import numpy as np
import sys
sys.path.append('/app/backend')

from universal_waveform_learning import UniversalWaveformLearningSystem

def test_plateau_detection():
    # Create the same square wave as in our test
    square_wave = np.array([5.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0, 3.0, 3.0, 
                           7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    
    print("Testing plateau detection on square wave:")
    print(f"Square wave data: {square_wave[:21]}")
    
    system = UniversalWaveformLearningSystem()
    
    # Test plateau detection
    plateaus = system._find_plateaus(square_wave)
    print(f"Found {len(plateaus)} plateaus:")
    for i, plateau in enumerate(plateaus):
        print(f"  Plateau {i+1}: start={plateau['start']}, length={plateau['length']}, value={plateau['value']}")
    
    # Test transitions
    transitions = system._find_sharp_transitions(square_wave)
    print(f"Found {len(transitions)} transitions:")
    for i, trans in enumerate(transitions):
        print(f"  Transition {i+1}: index={trans['index']}, magnitude={trans['magnitude']}, direction={trans['direction']}")
    
    # Test full square wave detection
    result = system._detect_square_wave_pattern(square_wave)
    print(f"Square wave detection result:")
    print(f"  Confidence: {result.get('confidence', 0.0)}")
    print(f"  Strength: {result.get('strength', 0.0)}")
    print(f"  Pattern type: {result.get('pattern_type', 'None')}")

if __name__ == "__main__":
    test_plateau_detection()