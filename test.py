import json
from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model("https://rr-traffic-video-stream.s3.us-east-1.amazonaws.com/2024/video-1713951019162.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIA2MJP5IYU7K4O4LPV%2F20250307%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250307T213130Z&X-Amz-Expires=600000&X-Amz-Signature=02cd2d77f76c4b75aaa1bbb871ef58387f90c601a41bb0af7d2d86b16c51c532&X-Amz-SignedHeaders=host&response-content-type=video%2Fmp4&x-id=GetObject", show=True)
cv2.waitKey()

accident={
  "description": {
    "generated": "In a two-lane road scenario, a truck is navigating a right-hand bend at an excessive speed, leading to instability of the container it is carrying. As the truck leans precariously, a nearby car attempts to overtake but promptly halts this maneuver upon observing the truck's precarious situation, demonstrating a reaction of defensive driving.\n\nThe situation escalates as the truck ultimately topples over due to the instability caused by the high speed on the curve. Concurrently, another vehicle experiences an unrelated incident on the same stretch of road, running off straight on the right-handed curve, indicative of a single vehicle accident, which could be related to similar factors such as speed or road conditions.",
    "manual": "",
    "prompt": "Could you furnish a concise, fact-based description of the given traffic scenario in one or two paragraphs? Please refrain from offering a summary paragraph. Take the time to thoroughly review the following text, as this information is important to me.\nThe truck is going around a bend in a two lane road too fast The container on the back of the truck becomes unstable and causes the truck to toppel over Before noticing that the truck is unstable the car wants to overtake the truck, however, it stops overtaking as an act of defensive driving when the driver notices the issues with the truck\nPersonal vehicle or car, \nA single vehicle accident where the vehicle \"runs off\" straight in a right handed curved road."
  },
  "verificationReport": {
    "generated": "# Comprehensive Verification Plan for Automated Vehicle Systems Involved in Two-Lane Road Incident\n\n## 1. Introduction\n\nThis comprehensive verification plan aims to investigate the automated vehicle (AV) systems in relation to an accident scenario involving a truck navigating a bend at excessive speed and a car attempting an overtaking manoeuvre. The verification process focuses on two critical systems of the AV: the perception system and the planning system. Our approach seeks to ensure the reliability, safety, and responsiveness of the AV against similar scenarios in the future.\n\n## 2. Objective of the Test\n\nThe primary objective of this test is to assess and validate the automated vehicle's perception and planning systems in scenarios reflective of the accident described. Specifically, the goals are to:\n\n- Evaluate the effectiveness of lane detection capabilities of the perception system under various environmental conditions.\n- Assess the decision-making processes of the planning system when confronted with dynamic and hazardous situations, including the need for defensive driving.\n- Ensure that safety measures are in place to prevent unforeseen accidents arising from similar conditions observed in the incident.\n\n## 3. Verification for the Perception System\n\nThis section outlines the verification steps for the perception system, classified into categories based on testing focus.\n\n### 3.1 Lane Detection\n\n1. **Verify Lane Marker Detection**\n   - Confirm that the perception system can accurately detect lane markers under various road conditions.\n   - Test with standard lane markers, double lines, and coloured markers to ensure reliable determination of lane boundaries.\n\n2. **Verify Width Between Lane Markers**\n   - Conduct tests to ensure that the measured width between left and right lane markers conforms to regulatory standards.\n\n3. **Verify Lane Centreing**\n   - Evaluate the ability of the vehicle to maintain its position within designated lanes during normal operations.\n\n### 3.2 Adversarial Conditions\n\n4. **Verify Detection in Adversarial Conditions (Rain, Fog, Glare)**\n   - Simulate adverse weather conditions to assess the lane detection system's robustness and reliability.\n\n5. **Verify Lane Marker Detection During Wiper Use**\n   - Test the perception system’s ability to identify lane markers during initial wiping after rainfall, particularly considering sunlight interference.\n\n6. **Verify Shadow Lines from Roadside Barriers**\n   - Examine how variations in light and shade affect the detection of lane markers in proximity to roadside barriers.\n\n### 3.3 Complex Lane Scenarios\n\n7. **Verify Intermediate and Washed-Out Lane Markers**\n   - Assess the system’s ability to detect and interpret faded or poorly maintained lane markers.\n\n8. **Verify Functional Safety Redundancy**\n   - Evaluate the effectiveness of a secondary sensor (e.g. LiDAR) for lane detection, ensuring redundancy in lane-keeping functionality.\n\n### 3.4 Lane Departure Warnings\n\n9. **Verify Lane Departure Warning System**\n   - Confirm that the vehicle's lane departure warning system triggers accurately when approaching lane boundaries, especially during high-speed scenarios.\n\n## 4. Verification for the Planning System\n\nThe planning system's verification focuses on assessing decision-making capabilities in complex, dynamic environments.\n\n### 4.1 Defensive Driving Behaviour\n\n10. **Verify Decision-Making in Dynamic Situations**\n    - Test the vehicle's ability to assess surrounding traffic conditions and determine when to execute defensive driving tactics, such as yielding to unstable vehicles.\n\n11. **Simulate Interaction with Precarious Vehicles**\n    - Create scenarios where the truck's instability presents a risk, assessing whether the AV correctly identifies threats and reacts accordingly.\n\n### 4.2 Reaction Time Analysis\n\n12. **Evaluate Reaction Time to Unstable Vehicle Situations**\n    - Simulate the accident scenario to determine if the AV can respond effectively within a critical time frame, referencing the 7.1 seconds available during the incident.\n\n### 4.3 Safety Measures and Protocols\n\n13. **Assess Emergency Braking and Evasion Protocols**\n    - Verify the capacity for the AV's autonomous emergency braking system to engage during imminent threats, considering high-speed movement and reaction times.\n\n14. **Evaluate Coordination Among Multiple Vehicles**\n    - Test the AV’s ability to interpret the actions of surrounding vehicles and plan movements accordingly, particularly during evasive actions.\n\n## 5. Conclusion\n\nThis verification plan outlines a comprehensive approach to assessing the automated vehicle's perception and planning systems in light of the incident involving a truck and a car on a two-lane road. Through a rigorous testing protocol, we aim to identify and mitigate potential hazards, ensuring that the automated vehicle is equipped to handle similar situations safely and effectively in the future. Each verification step serves to bolster confidence in the vehicle’s technology and adherence to safety regulations, thus contributing to overall road safety.",
    "manual": "",
    "prompt": "Please develop a comprehensive verification plan in the style of an academic report for the below accident for an automated vehicle describing (A) the objective of the test, (B) verification for the perception system, and (C) verification for the planning system. The accident is as follows: In a two-lane road scenario, a truck is navigating a right-hand bend at an excessive speed, leading to instability of the container it is carrying. As the truck leans precariously, a nearby car attempts to overtake but promptly halts this maneuver upon observing the truck's precarious situation, demonstrating a reaction of defensive driving.\n\nThe situation escalates as the truck ultimately topples over due to the instability caused by the high speed on the curve. Concurrently, another vehicle experiences an unrelated incident on the same stretch of road, running off straight on the right-handed curve, indicative of a single vehicle accident, which could be related to similar factors such as speed or road conditions. The victim had 7.1s to react and the accident severity was 3. The following are some ideas for verification steps based on the accident type: The Causer is Truck,  Motorcycle,  The Victim is Personal vehicle or car,  \"Verify lane marker detection [perception]. Verify that the width between the lane markers right and left is correct [perception].   Verify lane centering [perception].   Verify lane market detection in adversarial conditions as rain, fog, glare [perception]. Verify lane marker detection during smudge created by the first instance of wiper and sunlight [perception].   Verify shadow lines from road side barriers [perception] Verify intermediant and washed out lane markers [perception] Verify double and coloured lane markers [perception] Verify functional safety redundancy of a second sensor and/or lidar [perception] Lane Departure Warning System\". Additionally please categorise the verification steps and use UK spelling and grammar conventions."
  },
  "severity": -1,
  "migrationID": "5c0fce879258a4002d048f33",
  "migrationDate": 1539659832382,
  "accidentDate": 1539140580000,
  "location": {
    "type": "Feature",
    "properties": {},
    "geometry": {
      "type": "Point",
      "coordinates": [
        103.769755646352,
        1.41975014012237
      ]
    },
    "id": "5c0fce879258a4002d048f33"
  },
  "streetname": "Seletar Expressway, Sungei Kadut, Singapore 738078",
  "verification": "PREMIGRATION",
  "weather": [
    "6006"
  ],
  "time": [],
  "occlusion": [],
  "behaviour": [],
  "accidentTypes": [
    "013"
  ],
  "isComplex": True,
  "isEdge": True,
  "isChain": True,
  "archetype": "ARCHETYPE",
  "comment": "The truck is going around a bend in a two lane road too fast The container on the back of the truck becomes unstable and causes the truck to toppel over Before noticing that the truck is unstable the car wants to overtake the truck, however, it stops overtaking as an act of defensive driving when the driver notices the issues with the truck",
  "boundingBoxes": [],
  "scenario": {
    "timestamps": {
      "T0": 0,
      "T6": 27.7
    },
    "crashEnergy": 0,
    "impactForce": 0
  },
  "actors": [
    {
      "type": "TRUCK",
      "involvement": "CAUSER",
      "severity": 3,
      "timestamps": {
        "T1": -1,
        "T2": -1,
        "T3": -1,
        "T4": -1,
        "T5": -1
      },
      "scale": [
        1,
        1,
        1
      ],
      "trajectory": [
        {
          "frame": 0,
          "state": "KEYFRAME",
          "position": [
            103.7706180010095,
            1.419046343129378,
            0
          ],
          "rotation": [
            0,
            1.1336884446128115,
            0
          ]
        },
        {
          "frame": 1,
          "state": "INTERPOLATED",
          "position": [
            103.77060550111021,
            1.4190519194017592,
            0
          ],
          "rotation": [
            0,
            1.1336884446128115,
            0
          ]
        },
        {
          "frame": 2,
          "state": "INTERPOLATED",
          "position": [
            103.77059300121086,
            1.419057495674077,
            0
          ],
          "rotation": [
            0,
            1.1336884446128115,
            0
          ]
        },
        {
          "frame": 3,
          "state": "INTERPOLATED",
          "position": [
            103.77058050131144,
            1.419063071946327,
            0
          ],
          "rotation": [
            0,
            1.1336884446128115,
            0
          ]
        },
        {
          "frame": 4,
          "state": "INTERPOLATED",
          "position": [
            103.77056800141197,
            1.41906864821851,
            0
          ],
          "rotation": [
            0,
            1.1336884446128115,
            0
          ]
        },
        {
          "frame": 5,
          "state": "INTERPOLATED",
          "position": [
            103.77055550151245,
            1.4190742244906254,
            0
          ],
          "rotation": [
            0,
            1.1336884446128115,
            0
          ]
        },
        {
          "frame": 6,
          "state": "INTERPOLATED",
          "position": [
            103.77054300161285,
            1.419079800762673,
            0
          ],
          "rotation": [
            0,
            1.1336884446128115,
            0
          ]
        },
        {
          "frame": 7,
          "state": "INTERPOLATED",
          "position": [
            103.7705305017132,
            1.4190853770346532,
            0
          ],
          "rotation": [
            0,
            1.1336884446128115,
            0
          ]
        },
        {
          "frame": 8,
          "state": "INTERPOLATED",
          "position": [
            103.77051800181349,
            1.4190909533065659,
            0
          ],
          "rotation": [
            0,
            1.1336884446128115,
            0
          ]
        },
        {
          "frame": 9,
          "state": "KEYFRAME",
          "position": [
            103.7705055019137,
            1.419096529578411,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 10,
          "state": "INTERPOLATED",
          "position": [
            103.7704936548941,
            1.4191030433974696,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 11,
          "state": "INTERPOLATED",
          "position": [
            103.7704818078744,
            1.4191095572164638,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 12,
          "state": "INTERPOLATED",
          "position": [
            103.77046996085465,
            1.419116071035397,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 13,
          "state": "INTERPOLATED",
          "position": [
            103.77045811383483,
            1.41912258485427,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 14,
          "state": "INTERPOLATED",
          "position": [
            103.77044626681497,
            1.4191290986730816,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 15,
          "state": "INTERPOLATED",
          "position": [
            103.77043441979501,
            1.4191356124918328,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 16,
          "state": "INTERPOLATED",
          "position": [
            103.77042257277499,
            1.4191421263105237,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 17,
          "state": "INTERPOLATED",
          "position": [
            103.77041072575491,
            1.4191486401291538,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 18,
          "state": "INTERPOLATED",
          "position": [
            103.77039887873477,
            1.4191551539477232,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 19,
          "state": "INTERPOLATED",
          "position": [
            103.77038703171456,
            1.4191616677662322,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 20,
          "state": "INTERPOLATED",
          "position": [
            103.77037518469426,
            1.41916818158468,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 21,
          "state": "INTERPOLATED",
          "position": [
            103.77036333767393,
            1.4191746954030675,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 22,
          "state": "INTERPOLATED",
          "position": [
            103.77035149065351,
            1.4191812092213945,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 23,
          "state": "INTERPOLATED",
          "position": [
            103.77033964363302,
            1.4191877230396606,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 24,
          "state": "INTERPOLATED",
          "position": [
            103.77032779661246,
            1.4191942368578658,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 25,
          "state": "INTERPOLATED",
          "position": [
            103.77031594959186,
            1.4192007506760107,
            0
          ],
          "rotation": [
            0,
            1.1002393429138442,
            0
          ]
        },
        {
          "frame": 26,
          "state": "KEYFRAME",
          "position": [
            103.77030410257117,
            1.4192072644940952,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 27,
          "state": "INTERPOLATED",
          "position": [
            103.77029196413058,
            1.4192148508834916,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 28,
          "state": "INTERPOLATED",
          "position": [
            103.7702798256899,
            1.4192224372728262,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 29,
          "state": "INTERPOLATED",
          "position": [
            103.77026768724915,
            1.4192300236620967,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 30,
          "state": "INTERPOLATED",
          "position": [
            103.77025554880831,
            1.4192376100513038,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 31,
          "state": "INTERPOLATED",
          "position": [
            103.7702434103674,
            1.4192451964404476,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 32,
          "state": "INTERPOLATED",
          "position": [
            103.7702312719264,
            1.4192527828295274,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 33,
          "state": "INTERPOLATED",
          "position": [
            103.77021913348531,
            1.419260369218543,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 34,
          "state": "INTERPOLATED",
          "position": [
            103.77020699504418,
            1.4192679556074959,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 35,
          "state": "INTERPOLATED",
          "position": [
            103.77019485660293,
            1.4192755419963845,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 36,
          "state": "INTERPOLATED",
          "position": [
            103.77018271816162,
            1.4192831283852096,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 37,
          "state": "INTERPOLATED",
          "position": [
            103.77017057972023,
            1.4192907147739713,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 38,
          "state": "INTERPOLATED",
          "position": [
            103.77015844127875,
            1.4192983011626685,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 39,
          "state": "INTERPOLATED",
          "position": [
            103.7701463028372,
            1.4193058875513027,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 40,
          "state": "INTERPOLATED",
          "position": [
            103.77013416439557,
            1.4193134739398732,
            0
          ],
          "rotation": [
            0,
            0.9977475924680516,
            0
          ]
        },
        {
          "frame": 41,
          "state": "KEYFRAME",
          "position": [
            103.77012202595385,
            1.41932106032838,
            0
          ],
          "rotation": [
            0,
            0.9112871721664866,
            0
          ]
        },
        {
          "frame": 42,
          "state": "INTERPOLATED",
          "position": [
            103.77011100855792,
            1.4193297289660893,
            0
          ],
          "rotation": [
            0,
            0.9112871721664866,
            0
          ]
        },
        {
          "frame": 43,
          "state": "INTERPOLATED",
          "position": [
            103.77009999116191,
            1.4193383976037481,
            0
          ],
          "rotation": [
            0,
            0.9112871721664866,
            0
          ]
        },
        {
          "frame": 44,
          "state": "INTERPOLATED",
          "position": [
            103.77008897376581,
            1.419347066241355,
            0
          ],
          "rotation": [
            0,
            0.9112871721664866,
            0
          ]
        },
        {
          "frame": 45,
          "state": "INTERPOLATED",
          "position": [
            103.77007795636962,
            1.4193557348789094,
            0
          ],
          "rotation": [
            0,
            0.9112871721664866,
            0
          ]
        },
        {
          "frame": 46,
          "state": "INTERPOLATED",
          "position": [
            103.77006693897336,
            1.4193644035164106,
            0
          ],
          "rotation": [
            0,
            0.9112871721664866,
            0
          ]
        },
        {
          "frame": 47,
          "state": "INTERPOLATED",
          "position": [
            103.77005592157701,
            1.4193730721538602,
            0
          ],
          "rotation": [
            0,
            0.9112871721664866,
            0
          ]
        },
        {
          "frame": 48,
          "state": "INTERPOLATED",
          "position": [
            103.77004490418058,
            1.4193817407912566,
            0
          ],
          "rotation": [
            0,
            0.9112871721664866,
            0
          ]
        },
        {
          "frame": 49,
          "state": "INTERPOLATED",
          "position": [
            103.77003388678406,
            1.4193904094286012,
            0
          ],
          "rotation": [
            0,
            0.9112871721664866,
            0
          ]
        },
        {
          "frame": 50,
          "state": "INTERPOLATED",
          "position": [
            103.77002286938746,
            1.4193990780658927,
            0
          ],
          "rotation": [
            0,
            0.9112871721664866,
            0
          ]
        },
        {
          "frame": 51,
          "state": "KEYFRAME",
          "position": [
            103.77001185199079,
            1.4194077467031323,
            0
          ],
          "rotation": [
            0,
            0.8081765791109551,
            0
          ]
        },
        {
          "frame": 52,
          "state": "INTERPOLATED",
          "position": [
            103.76999988660098,
            1.419420215686997,
            0
          ],
          "rotation": [
            0,
            0.8081765791109551,
            0
          ]
        },
        {
          "frame": 53,
          "state": "INTERPOLATED",
          "position": [
            103.76998792121104,
            1.419432684670799,
            0
          ],
          "rotation": [
            0,
            0.8081765791109551,
            0
          ]
        },
        {
          "frame": 54,
          "state": "INTERPOLATED",
          "position": [
            103.76997595582098,
            1.419445153654539,
            0
          ],
          "rotation": [
            0,
            0.8081765791109551,
            0
          ]
        },
        {
          "frame": 55,
          "state": "INTERPOLATED",
          "position": [
            103.76996399043078,
            1.4194576226382172,
            0
          ],
          "rotation": [
            0,
            0.8081765791109551,
            0
          ]
        },
        {
          "frame": 56,
          "state": "INTERPOLATED",
          "position": [
            103.76995202504047,
            1.4194700916218332,
            0
          ],
          "rotation": [
            0,
            0.8081765791109551,
            0
          ]
        },
        {
          "frame": 57,
          "state": "INTERPOLATED",
          "position": [
            103.76994005965001,
            1.4194825606053874,
            0
          ],
          "rotation": [
            0,
            0.8081765791109551,
            0
          ]
        },
        {
          "frame": 58,
          "state": "KEYFRAME",
          "position": [
            103.76992809425943,
            1.4194950295888797,
            0
          ],
          "rotation": [
            0,
            0.685907948003905,
            0
          ]
        },
        {
          "frame": 59,
          "state": "INTERPOLATED",
          "position": [
            103.76991775274388,
            1.4195090662490935,
            0
          ],
          "rotation": [
            0,
            0.685907948003905,
            0
          ]
        },
        {
          "frame": 60,
          "state": "INTERPOLATED",
          "position": [
            103.76990741122822,
            1.4195231029092616,
            0
          ],
          "rotation": [
            0,
            0.685907948003905,
            0
          ]
        },
        {
          "frame": 61,
          "state": "INTERPOLATED",
          "position": [
            103.76989706971241,
            1.4195371395693837,
            0
          ],
          "rotation": [
            0,
            0.685907948003905,
            0
          ]
        },
        {
          "frame": 62,
          "state": "INTERPOLATED",
          "position": [
            103.76988672819648,
            1.4195511762294593,
            0
          ],
          "rotation": [
            0,
            0.685907948003905,
            0
          ]
        },
        {
          "frame": 63,
          "state": "INTERPOLATED",
          "position": [
            103.76987638668042,
            1.419565212889489,
            0
          ],
          "rotation": [
            0,
            0.685907948003905,
            0
          ]
        },
        {
          "frame": 64,
          "state": "INTERPOLATED",
          "position": [
            103.76986604516426,
            1.4195792495494721,
            0
          ],
          "rotation": [
            0,
            0.685907948003905,
            0
          ]
        },
        {
          "frame": 65,
          "state": "INTERPOLATED",
          "position": [
            103.76985570364795,
            1.4195932862094092,
            0
          ],
          "rotation": [
            0,
            0.685907948003905,
            0
          ]
        },
        {
          "frame": 66,
          "state": "KEYFRAME",
          "position": [
            103.76984536213152,
            1.4196073228693002,
            0
          ],
          "rotation": [
            0.017046152979557385,
            0.37807920341832674,
            0.15953084180117505
          ]
        },
        {
          "frame": 67,
          "state": "INTERPOLATED",
          "position": [
            103.76983926863471,
            1.4196239521354381,
            0
          ],
          "rotation": [
            0.017046152979557385,
            0.37807920341832674,
            0.15953084180117505
          ]
        },
        {
          "frame": 68,
          "state": "INTERPOLATED",
          "position": [
            103.76983317513782,
            1.4196405814015598,
            0
          ],
          "rotation": [
            0.017046152979557385,
            0.37807920341832674,
            0.15953084180117505
          ]
        },
        {
          "frame": 69,
          "state": "INTERPOLATED",
          "position": [
            103.76982708164081,
            1.4196572106676655,
            0
          ],
          "rotation": [
            0.017046152979557385,
            0.37807920341832674,
            0.15953084180117505
          ]
        },
        {
          "frame": 70,
          "state": "INTERPOLATED",
          "position": [
            103.76982098814375,
            1.419673839933755,
            0
          ],
          "rotation": [
            0.017046152979557385,
            0.37807920341832674,
            0.15953084180117505
          ]
        },
        {
          "frame": 71,
          "state": "INTERPOLATED",
          "position": [
            103.76981489464657,
            1.4196904691998289,
            0
          ],
          "rotation": [
            0.017046152979557385,
            0.37807920341832674,
            0.15953084180117505
          ]
        },
        {
          "frame": 72,
          "state": "INTERPOLATED",
          "position": [
            103.76980880114932,
            1.4197070984658862,
            0
          ],
          "rotation": [
            0.017046152979557385,
            0.37807920341832674,
            0.15953084180117505
          ]
        },
        {
          "frame": 73,
          "state": "KEYFRAME",
          "position": [
            103.76980270765198,
            1.4197237277319277,
            0
          ],
          "rotation": [
            0.028811702078534115,
            0.30619808827308687,
            0.3246586099086756
          ]
        },
        {
          "frame": 74,
          "state": "INTERPOLATED",
          "position": [
            103.76979842361067,
            1.4197403569979616,
            0
          ],
          "rotation": [
            0.028811702078534115,
            0.30619808827308687,
            0.3246586099086756
          ]
        },
        {
          "frame": 75,
          "state": "INTERPOLATED",
          "position": [
            103.7697941395693,
            1.419756986263987,
            0
          ],
          "rotation": [
            0.028811702078534115,
            0.30619808827308687,
            0.3246586099086756
          ]
        },
        {
          "frame": 76,
          "state": "INTERPOLATED",
          "position": [
            103.76978985552788,
            1.4197736155300038,
            0
          ],
          "rotation": [
            0.028811702078534115,
            0.30619808827308687,
            0.3246586099086756
          ]
        },
        {
          "frame": 77,
          "state": "INTERPOLATED",
          "position": [
            103.76978557148638,
            1.4197902447960131,
            0
          ],
          "rotation": [
            0.028811702078534115,
            0.30619808827308687,
            0.3246586099086756
          ]
        },
        {
          "frame": 78,
          "state": "INTERPOLATED",
          "position": [
            103.76978128744483,
            1.4198068740620142,
            0
          ],
          "rotation": [
            0.028811702078534115,
            0.30619808827308687,
            0.3246586099086756
          ]
        },
        {
          "frame": 79,
          "state": "INTERPOLATED",
          "position": [
            103.7697770034032,
            1.4198235033280078,
            0
          ],
          "rotation": [
            0.028811702078534115,
            0.30619808827308687,
            0.3246586099086756
          ]
        },
        {
          "frame": 80,
          "state": "INTERPOLATED",
          "position": [
            103.76977271936153,
            1.4198401325939933,
            0
          ],
          "rotation": [
            0.028811702078534115,
            0.30619808827308687,
            0.3246586099086756
          ]
        },
        {
          "frame": 81,
          "state": "KEYFRAME",
          "position": [
            103.76976843531979,
            1.4198567618599707,
            0
          ],
          "rotation": [
            0.09941485384422946,
            -0.014278063120211876,
            0.3961960480068331
          ]
        },
        {
          "frame": 82,
          "state": "INTERPOLATED",
          "position": [
            103.76976876842853,
            1.4198707422906507,
            0
          ],
          "rotation": [
            0.09941485384422946,
            -0.014278063120211876,
            0.3961960480068331
          ]
        },
        {
          "frame": 83,
          "state": "INTERPOLATED",
          "position": [
            103.76976910153728,
            1.4198847227213303,
            0
          ],
          "rotation": [
            0.09941485384422946,
            -0.014278063120211876,
            0.3961960480068331
          ]
        },
        {
          "frame": 84,
          "state": "INTERPOLATED",
          "position": [
            103.76976943464602,
            1.41989870315201,
            0
          ],
          "rotation": [
            0.09941485384422946,
            -0.014278063120211876,
            0.3961960480068331
          ]
        },
        {
          "frame": 85,
          "state": "INTERPOLATED",
          "position": [
            103.76976976775478,
            1.4199126835826898,
            0
          ],
          "rotation": [
            0.09941485384422946,
            -0.014278063120211876,
            0.3961960480068331
          ]
        },
        {
          "frame": 86,
          "state": "INTERPOLATED",
          "position": [
            103.76977010086354,
            1.4199266640133694,
            0
          ],
          "rotation": [
            0.09941485384422946,
            -0.014278063120211876,
            0.3961960480068331
          ]
        },
        {
          "frame": 87,
          "state": "INTERPOLATED",
          "position": [
            103.76977043397231,
            1.4199406444440492,
            0
          ],
          "rotation": [
            0.09941485384422946,
            -0.014278063120211876,
            0.3961960480068331
          ]
        },
        {
          "frame": 88,
          "state": "INTERPOLATED",
          "position": [
            103.76977076708108,
            1.4199546248747286,
            0
          ],
          "rotation": [
            0.09941485384422946,
            -0.014278063120211876,
            0.3961960480068331
          ]
        },
        {
          "frame": 89,
          "state": "KEYFRAME",
          "position": [
            103.76977110018984,
            1.4199686053054081,
            0
          ],
          "rotation": [
            0.10415090827112146,
            -0.025271980644177964,
            0.49340177265429597
          ]
        },
        {
          "frame": 90,
          "state": "INTERPOLATED",
          "position": [
            103.76977274636705,
            1.419982585736089,
            0
          ],
          "rotation": [
            0.10415090827112146,
            -0.025271980644177964,
            0.49340177265429597
          ]
        },
        {
          "frame": 91,
          "state": "INTERPOLATED",
          "position": [
            103.76977439254428,
            1.4199965661667686,
            0
          ],
          "rotation": [
            0.10415090827112146,
            -0.025271980644177964,
            0.49340177265429597
          ]
        },
        {
          "frame": 92,
          "state": "INTERPOLATED",
          "position": [
            103.76977603872152,
            1.4200105465974466,
            0
          ],
          "rotation": [
            0.10415090827112146,
            -0.025271980644177964,
            0.49340177265429597
          ]
        },
        {
          "frame": 93,
          "state": "INTERPOLATED",
          "position": [
            103.76977768489878,
            1.4200245270281235,
            0
          ],
          "rotation": [
            0.10415090827112146,
            -0.025271980644177964,
            0.49340177265429597
          ]
        },
        {
          "frame": 94,
          "state": "INTERPOLATED",
          "position": [
            103.76977933107608,
            1.4200385074587996,
            0
          ],
          "rotation": [
            0.10415090827112146,
            -0.025271980644177964,
            0.49340177265429597
          ]
        },
        {
          "frame": 95,
          "state": "INTERPOLATED",
          "position": [
            103.76978097725338,
            1.4200524878894742,
            0
          ],
          "rotation": [
            0.10415090827112146,
            -0.025271980644177964,
            0.49340177265429597
          ]
        },
        {
          "frame": 96,
          "state": "INTERPOLATED",
          "position": [
            103.7697826234307,
            1.4200664683201478,
            0
          ],
          "rotation": [
            0.10415090827112146,
            -0.025271980644177964,
            0.49340177265429597
          ]
        },
        {
          "frame": 97,
          "state": "KEYFRAME",
          "position": [
            103.76978426960804,
            1.42008044875082,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 98,
          "state": "INTERPOLATED",
          "position": [
            103.76978545754359,
            1.4200856527593795,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 99,
          "state": "INTERPOLATED",
          "position": [
            103.76978664547913,
            1.420090856767938,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 100,
          "state": "INTERPOLATED",
          "position": [
            103.7697878334147,
            1.4200960607764963,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 101,
          "state": "INTERPOLATED",
          "position": [
            103.76978902135025,
            1.4201012647850537,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 102,
          "state": "INTERPOLATED",
          "position": [
            103.7697902092858,
            1.4201064687936107,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 103,
          "state": "INTERPOLATED",
          "position": [
            103.76979139722138,
            1.4201116728021668,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 104,
          "state": "INTERPOLATED",
          "position": [
            103.76979258515695,
            1.4201168768107224,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 105,
          "state": "INTERPOLATED",
          "position": [
            103.76979377309253,
            1.4201220808192774,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 106,
          "state": "INTERPOLATED",
          "position": [
            103.76979496102811,
            1.420127284827832,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 107,
          "state": "INTERPOLATED",
          "position": [
            103.7697961489637,
            1.4201324888363855,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 108,
          "state": "INTERPOLATED",
          "position": [
            103.7697973368993,
            1.420137692844939,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 109,
          "state": "INTERPOLATED",
          "position": [
            103.7697985248349,
            1.4201428968534913,
            0
          ],
          "rotation": [
            0.15349099124021304,
            -0.32755173877700344,
            0.5789879460854117
          ]
        },
        {
          "frame": 110,
          "state": "KEYFRAME",
          "position": [
            103.7697997127705,
            1.4201481008620433,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 111,
          "state": "INTERPOLATED",
          "position": [
            103.76979980020646,
            1.4201485351672969,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 112,
          "state": "INTERPOLATED",
          "position": [
            103.7697998876424,
            1.4201489694725515,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 113,
          "state": "INTERPOLATED",
          "position": [
            103.76979997507836,
            1.420149403777806,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 114,
          "state": "INTERPOLATED",
          "position": [
            103.76980006251429,
            1.4201498380830604,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 115,
          "state": "INTERPOLATED",
          "position": [
            103.76980014995023,
            1.4201502723883153,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 116,
          "state": "INTERPOLATED",
          "position": [
            103.76980023738619,
            1.4201507066935697,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 117,
          "state": "INTERPOLATED",
          "position": [
            103.76980032482213,
            1.4201511409988241,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 118,
          "state": "INTERPOLATED",
          "position": [
            103.76980041225808,
            1.4201515753040739,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 119,
          "state": "INTERPOLATED",
          "position": [
            103.76980049969403,
            1.4201520096093332,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 120,
          "state": "INTERPOLATED",
          "position": [
            103.76980058712996,
            1.4201524439145878,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 121,
          "state": "INTERPOLATED",
          "position": [
            103.76980067456591,
            1.4201528782198425,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 122,
          "state": "INTERPOLATED",
          "position": [
            103.76980076200185,
            1.420153312525097,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 123,
          "state": "INTERPOLATED",
          "position": [
            103.7698008494378,
            1.4201537468303511,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 124,
          "state": "INTERPOLATED",
          "position": [
            103.76980093687375,
            1.4201541811356055,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 125,
          "state": "INTERPOLATED",
          "position": [
            103.76980102430969,
            1.42015461544086,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 126,
          "state": "INTERPOLATED",
          "position": [
            103.76980111174565,
            1.4201550497461146,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 127,
          "state": "INTERPOLATED",
          "position": [
            103.76980119918159,
            1.420155484051369,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 128,
          "state": "INTERPOLATED",
          "position": [
            103.76980128661752,
            1.4201559183566237,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 129,
          "state": "INTERPOLATED",
          "position": [
            103.76980137405349,
            1.4201563526618781,
            -5.78704646355827e-14
          ],
          "rotation": [
            0.4994890108635981,
            -0.4745051037764062,
            1.029128601665684
          ]
        },
        {
          "frame": 130,
          "state": "KEYFRAME",
          "position": [
            103.76980146148942,
            1.4201567869671325,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 131,
          "state": "INTERPOLATED",
          "position": [
            103.76980146174115,
            1.4201568032478944,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 132,
          "state": "INTERPOLATED",
          "position": [
            103.76980146199288,
            1.4201568195286562,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 133,
          "state": "INTERPOLATED",
          "position": [
            103.76980146224462,
            1.420156835809418,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 134,
          "state": "INTERPOLATED",
          "position": [
            103.76980146249633,
            1.4201568520901802,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 135,
          "state": "INTERPOLATED",
          "position": [
            103.76980146274808,
            1.420156868370942,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 136,
          "state": "INTERPOLATED",
          "position": [
            103.7698014629998,
            1.4201568846517043,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 137,
          "state": "INTERPOLATED",
          "position": [
            103.76980146325153,
            1.4201569009324662,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 138,
          "state": "INTERPOLATED",
          "position": [
            103.76980146350324,
            1.4201569172132282,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 139,
          "state": "INTERPOLATED",
          "position": [
            103.76980146375499,
            1.42015693349399,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 140,
          "state": "INTERPOLATED",
          "position": [
            103.76980146400672,
            1.4201569497747522,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 141,
          "state": "INTERPOLATED",
          "position": [
            103.76980146425844,
            1.420156966055514,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 142,
          "state": "INTERPOLATED",
          "position": [
            103.76980146451018,
            1.420156982336276,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 143,
          "state": "INTERPOLATED",
          "position": [
            103.7698014647619,
            1.420156998617038,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 144,
          "state": "INTERPOLATED",
          "position": [
            103.76980146501363,
            1.4201570148977998,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 145,
          "state": "INTERPOLATED",
          "position": [
            103.76980146526535,
            1.4201570311785618,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 146,
          "state": "INTERPOLATED",
          "position": [
            103.76980146551709,
            1.4201570474593237,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 147,
          "state": "INTERPOLATED",
          "position": [
            103.76980146576882,
            1.420157063740086,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 148,
          "state": "INTERPOLATED",
          "position": [
            103.76980146602054,
            1.4201570800208478,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 149,
          "state": "INTERPOLATED",
          "position": [
            103.76980146627228,
            1.42015709630161,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 150,
          "state": "INTERPOLATED",
          "position": [
            103.769801466524,
            1.4201571125823718,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 151,
          "state": "INTERPOLATED",
          "position": [
            103.76980146677573,
            1.4201571288631338,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 152,
          "state": "INTERPOLATED",
          "position": [
            103.76980146702746,
            1.4201571451438957,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 153,
          "state": "INTERPOLATED",
          "position": [
            103.7698014672792,
            1.4201571614246575,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 154,
          "state": "INTERPOLATED",
          "position": [
            103.76980146753091,
            1.4201571777054194,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 155,
          "state": "INTERPOLATED",
          "position": [
            103.76980146778264,
            1.4201571939861815,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 156,
          "state": "INTERPOLATED",
          "position": [
            103.76980146803439,
            1.4201572102669433,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 157,
          "state": "INTERPOLATED",
          "position": [
            103.7698014682861,
            1.4201572265477052,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 158,
          "state": "INTERPOLATED",
          "position": [
            103.76980146853784,
            1.4201572428284672,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 159,
          "state": "INTERPOLATED",
          "position": [
            103.76980146878957,
            1.420157259109229,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 160,
          "state": "INTERPOLATED",
          "position": [
            103.7698014690413,
            1.4201572753899911,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 161,
          "state": "INTERPOLATED",
          "position": [
            103.76980146929301,
            1.4201572916707532,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 162,
          "state": "INTERPOLATED",
          "position": [
            103.76980146954475,
            1.420157307951515,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 163,
          "state": "INTERPOLATED",
          "position": [
            103.76980146979649,
            1.4201573242322771,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 164,
          "state": "INTERPOLATED",
          "position": [
            103.76980147004821,
            1.420157340513039,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 165,
          "state": "INTERPOLATED",
          "position": [
            103.76980147029994,
            1.420157356793801,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 166,
          "state": "INTERPOLATED",
          "position": [
            103.76980147055167,
            1.420157373074563,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 167,
          "state": "INTERPOLATED",
          "position": [
            103.7698014708034,
            1.420157389355325,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 168,
          "state": "INTERPOLATED",
          "position": [
            103.76980147105512,
            1.4201574056360868,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 169,
          "state": "INTERPOLATED",
          "position": [
            103.76980147130685,
            1.420157421916849,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 170,
          "state": "INTERPOLATED",
          "position": [
            103.76980147155858,
            1.4201574381976108,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 171,
          "state": "INTERPOLATED",
          "position": [
            103.76980147181031,
            1.4201574544783728,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 172,
          "state": "INTERPOLATED",
          "position": [
            103.76980147206204,
            1.4201574707591347,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 173,
          "state": "INTERPOLATED",
          "position": [
            103.76980147231377,
            1.4201574870398967,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 174,
          "state": "INTERPOLATED",
          "position": [
            103.7698014725655,
            1.4201575033206586,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 175,
          "state": "INTERPOLATED",
          "position": [
            103.76980147281722,
            1.4201575196014207,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 176,
          "state": "INTERPOLATED",
          "position": [
            103.76980147306895,
            1.4201575358821825,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 177,
          "state": "INTERPOLATED",
          "position": [
            103.76980147332068,
            1.4201575521629446,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 178,
          "state": "INTERPOLATED",
          "position": [
            103.76980147357241,
            1.4201575684437062,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 179,
          "state": "INTERPOLATED",
          "position": [
            103.76980147382415,
            1.4201575847244683,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 180,
          "state": "INTERPOLATED",
          "position": [
            103.76980147407588,
            1.4201576010052301,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 181,
          "state": "INTERPOLATED",
          "position": [
            103.76980147432761,
            1.4201576172859922,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 182,
          "state": "INTERPOLATED",
          "position": [
            103.76980147457932,
            1.420157633566754,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 183,
          "state": "INTERPOLATED",
          "position": [
            103.76980147483106,
            1.420157649847516,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 184,
          "state": "INTERPOLATED",
          "position": [
            103.76980147508279,
            1.420157666128278,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 185,
          "state": "INTERPOLATED",
          "position": [
            103.76980147533452,
            1.4201576824090398,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 186,
          "state": "INTERPOLATED",
          "position": [
            103.76980147558623,
            1.420157698689802,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 187,
          "state": "INTERPOLATED",
          "position": [
            103.76980147583798,
            1.4201577149705638,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 188,
          "state": "INTERPOLATED",
          "position": [
            103.76980147608971,
            1.4201577312513258,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 189,
          "state": "INTERPOLATED",
          "position": [
            103.76980147634143,
            1.420157747532088,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 190,
          "state": "INTERPOLATED",
          "position": [
            103.76980147659316,
            1.4201577638128497,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 191,
          "state": "INTERPOLATED",
          "position": [
            103.76980147684489,
            1.4201577800936118,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 192,
          "state": "INTERPOLATED",
          "position": [
            103.76980147709662,
            1.4201577963743737,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 193,
          "state": "INTERPOLATED",
          "position": [
            103.76980147734834,
            1.4201578126551357,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 194,
          "state": "INTERPOLATED",
          "position": [
            103.76980147760008,
            1.4201578289358978,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 195,
          "state": "INTERPOLATED",
          "position": [
            103.76980147785181,
            1.4201578452166599,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 196,
          "state": "INTERPOLATED",
          "position": [
            103.76980147810353,
            1.4201578614974217,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 197,
          "state": "INTERPOLATED",
          "position": [
            103.76980147835526,
            1.4201578777781836,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 198,
          "state": "INTERPOLATED",
          "position": [
            103.769801478607,
            1.4201578940589457,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 199,
          "state": "INTERPOLATED",
          "position": [
            103.76980147885872,
            1.4201579103397075,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 200,
          "state": "INTERPOLATED",
          "position": [
            103.76980147911044,
            1.4201579266204694,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 201,
          "state": "INTERPOLATED",
          "position": [
            103.76980147936219,
            1.4201579429012312,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 202,
          "state": "INTERPOLATED",
          "position": [
            103.7698014796139,
            1.4201579591819933,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 203,
          "state": "INTERPOLATED",
          "position": [
            103.76980147986563,
            1.4201579754627551,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 204,
          "state": "INTERPOLATED",
          "position": [
            103.76980148011737,
            1.420157991743517,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 205,
          "state": "INTERPOLATED",
          "position": [
            103.7698014803691,
            1.420158008024279,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 206,
          "state": "INTERPOLATED",
          "position": [
            103.76980148062083,
            1.420158024305041,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 207,
          "state": "INTERPOLATED",
          "position": [
            103.76980148087254,
            1.420158040585803,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 208,
          "state": "INTERPOLATED",
          "position": [
            103.76980148112429,
            1.4201580568665648,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 209,
          "state": "INTERPOLATED",
          "position": [
            103.769801481376,
            1.420158073147327,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 210,
          "state": "INTERPOLATED",
          "position": [
            103.76980148162774,
            1.4201580894280887,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 211,
          "state": "INTERPOLATED",
          "position": [
            103.76980148187948,
            1.4201581057088508,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 212,
          "state": "INTERPOLATED",
          "position": [
            103.7698014821312,
            1.4201581219896127,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 213,
          "state": "INTERPOLATED",
          "position": [
            103.76980148238293,
            1.4201581382703747,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 214,
          "state": "INTERPOLATED",
          "position": [
            103.76980148263465,
            1.4201581545511366,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 215,
          "state": "INTERPOLATED",
          "position": [
            103.7698014828864,
            1.4201581708318987,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 216,
          "state": "INTERPOLATED",
          "position": [
            103.76980148313811,
            1.4201581871126605,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 217,
          "state": "INTERPOLATED",
          "position": [
            103.76980148338984,
            1.4201582033934226,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 218,
          "state": "INTERPOLATED",
          "position": [
            103.76980148364157,
            1.4201582196741844,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 219,
          "state": "INTERPOLATED",
          "position": [
            103.7698014838933,
            1.4201582359549465,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 220,
          "state": "INTERPOLATED",
          "position": [
            103.76980148414503,
            1.4201582522357084,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 221,
          "state": "INTERPOLATED",
          "position": [
            103.76980148439675,
            1.4201582685164706,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 222,
          "state": "INTERPOLATED",
          "position": [
            103.7698014846485,
            1.4201582847972325,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 223,
          "state": "INTERPOLATED",
          "position": [
            103.76980148490021,
            1.4201583010779946,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 224,
          "state": "INTERPOLATED",
          "position": [
            103.76980148515194,
            1.4201583173587562,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 225,
          "state": "INTERPOLATED",
          "position": [
            103.76980148540368,
            1.4201583336395183,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 226,
          "state": "INTERPOLATED",
          "position": [
            103.7698014856554,
            1.4201583499202801,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 227,
          "state": "INTERPOLATED",
          "position": [
            103.76980148590714,
            1.4201583662010422,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 228,
          "state": "INTERPOLATED",
          "position": [
            103.76980148615885,
            1.4201583824818043,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 229,
          "state": "INTERPOLATED",
          "position": [
            103.7698014864106,
            1.420158398762566,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 230,
          "state": "INTERPOLATED",
          "position": [
            103.76980148666232,
            1.4201584150433282,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 231,
          "state": "INTERPOLATED",
          "position": [
            103.76980148691405,
            1.42015843132409,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 232,
          "state": "INTERPOLATED",
          "position": [
            103.76980148716578,
            1.420158447604852,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 233,
          "state": "INTERPOLATED",
          "position": [
            103.76980148741751,
            1.420158463885614,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 234,
          "state": "INTERPOLATED",
          "position": [
            103.76980148766923,
            1.420158480166376,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 235,
          "state": "INTERPOLATED",
          "position": [
            103.76980148792097,
            1.4201584964471379,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 236,
          "state": "INTERPOLATED",
          "position": [
            103.7698014881727,
            1.4201585127279,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 237,
          "state": "INTERPOLATED",
          "position": [
            103.76980148842442,
            1.4201585290086618,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 238,
          "state": "INTERPOLATED",
          "position": [
            103.76980148867615,
            1.4201585452894239,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 239,
          "state": "INTERPOLATED",
          "position": [
            103.76980148892788,
            1.4201585615701857,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 240,
          "state": "INTERPOLATED",
          "position": [
            103.76980148917961,
            1.4201585778509478,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 241,
          "state": "INTERPOLATED",
          "position": [
            103.76980148943133,
            1.4201585941317096,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 242,
          "state": "INTERPOLATED",
          "position": [
            103.76980148968308,
            1.4201586104124717,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 243,
          "state": "INTERPOLATED",
          "position": [
            103.7698014899348,
            1.4201586266932336,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 244,
          "state": "INTERPOLATED",
          "position": [
            103.76980149018652,
            1.4201586429739954,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 245,
          "state": "INTERPOLATED",
          "position": [
            103.76980149043825,
            1.4201586592547575,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 246,
          "state": "INTERPOLATED",
          "position": [
            103.76980149068999,
            1.4201586755355193,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 247,
          "state": "INTERPOLATED",
          "position": [
            103.76980149094172,
            1.4201586918162812,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 248,
          "state": "INTERPOLATED",
          "position": [
            103.76980149119343,
            1.420158708097043,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 249,
          "state": "INTERPOLATED",
          "position": [
            103.76980149144518,
            1.420158724377805,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 250,
          "state": "INTERPOLATED",
          "position": [
            103.7698014916969,
            1.420158740658567,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 251,
          "state": "INTERPOLATED",
          "position": [
            103.76980149194863,
            1.420158756939329,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 252,
          "state": "INTERPOLATED",
          "position": [
            103.76980149220036,
            1.4201587732200909,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 253,
          "state": "INTERPOLATED",
          "position": [
            103.76980149245209,
            1.420158789500853,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 254,
          "state": "INTERPOLATED",
          "position": [
            103.76980149270382,
            1.4201588057816148,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 255,
          "state": "INTERPOLATED",
          "position": [
            103.76980149295554,
            1.4201588220623769,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 256,
          "state": "INTERPOLATED",
          "position": [
            103.76980149320728,
            1.4201588383431387,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 257,
          "state": "INTERPOLATED",
          "position": [
            103.769801493459,
            1.4201588546239008,
            -5.787046513346264e-14
          ],
          "rotation": [
            1.2257564993814412,
            -1.3087347639508582,
            1.2936104266299737
          ]
        },
        {
          "frame": 258,
          "state": "KEYFRAME",
          "position": [
            103.76980149371073,
            1.4201588709046629,
            -5.78704649382032e-14
          ],
          "rotation": [
            1.1164382774433612,
            -1.1604461448786276,
            1.2148064109140067
          ]
        }
      ]
    },
    {
      "type": "CAR",
      "involvement": "VICTIM",
      "severity": 2.5,
      "timestamps": {
        "T1": 6.9,
        "T2": 11.3,
        "T3": 14,
        "T4": 15.6,
        "T5": 16.6
      },
      "scale": [
        1,
        1,
        1
      ],
      "trajectory": [
        {
          "frame": 0,
          "state": "KEYFRAME",
          "position": [
            103.77074028254292,
            1.4190208088893466,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 1,
          "state": "INTERPOLATED",
          "position": [
            103.77072879741513,
            1.4190263945926003,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 2,
          "state": "INTERPOLATED",
          "position": [
            103.7707173122873,
            1.4190319802958002,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 3,
          "state": "INTERPOLATED",
          "position": [
            103.7707058271594,
            1.4190375659989425,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 4,
          "state": "INTERPOLATED",
          "position": [
            103.77069434203146,
            1.4190431517020283,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 5,
          "state": "INTERPOLATED",
          "position": [
            103.77068285690345,
            1.4190487374050564,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 6,
          "state": "INTERPOLATED",
          "position": [
            103.77067137177538,
            1.4190543231080284,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 7,
          "state": "INTERPOLATED",
          "position": [
            103.77065988664727,
            1.419059908810943,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 8,
          "state": "INTERPOLATED",
          "position": [
            103.7706484015191,
            1.4190654945138004,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 9,
          "state": "INTERPOLATED",
          "position": [
            103.77063691639087,
            1.4190710802166009,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 10,
          "state": "INTERPOLATED",
          "position": [
            103.77062543126259,
            1.4190766659193446,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 11,
          "state": "INTERPOLATED",
          "position": [
            103.77061394613425,
            1.419082251622031,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 12,
          "state": "INTERPOLATED",
          "position": [
            103.77060246100585,
            1.4190878373246603,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 13,
          "state": "INTERPOLATED",
          "position": [
            103.77059097587741,
            1.419093423027233,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 14,
          "state": "INTERPOLATED",
          "position": [
            103.7705794907489,
            1.4190990087297488,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 15,
          "state": "INTERPOLATED",
          "position": [
            103.77056800562033,
            1.4191045944322072,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 16,
          "state": "INTERPOLATED",
          "position": [
            103.77055652049172,
            1.4191101801346089,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 17,
          "state": "INTERPOLATED",
          "position": [
            103.77054503536304,
            1.4191157658369535,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 18,
          "state": "INTERPOLATED",
          "position": [
            103.77053355023432,
            1.4191213515392407,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 19,
          "state": "INTERPOLATED",
          "position": [
            103.77052206510555,
            1.4191269372414714,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 20,
          "state": "INTERPOLATED",
          "position": [
            103.7705105799767,
            1.4191325229436451,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 21,
          "state": "INTERPOLATED",
          "position": [
            103.7704990948478,
            1.4191381086457615,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 22,
          "state": "INTERPOLATED",
          "position": [
            103.77048760971886,
            1.4191436943478213,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 23,
          "state": "INTERPOLATED",
          "position": [
            103.77047612458986,
            1.4191492800498238,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 24,
          "state": "INTERPOLATED",
          "position": [
            103.7704646394608,
            1.4191548657517694,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 25,
          "state": "INTERPOLATED",
          "position": [
            103.77045315433169,
            1.419160451453658,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 26,
          "state": "INTERPOLATED",
          "position": [
            103.77044166920251,
            1.4191660371554895,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 27,
          "state": "INTERPOLATED",
          "position": [
            103.77043018407329,
            1.419171622857264,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 28,
          "state": "INTERPOLATED",
          "position": [
            103.77041869894401,
            1.4191772085589816,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 29,
          "state": "INTERPOLATED",
          "position": [
            103.77040721381468,
            1.4191827942606419,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 30,
          "state": "INTERPOLATED",
          "position": [
            103.77039572868527,
            1.4191883799622458,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 31,
          "state": "INTERPOLATED",
          "position": [
            103.77038424355584,
            1.4191939656637922,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 32,
          "state": "INTERPOLATED",
          "position": [
            103.77037275842633,
            1.4191995513652818,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 33,
          "state": "INTERPOLATED",
          "position": [
            103.77036127329677,
            1.4192051370667143,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 34,
          "state": "INTERPOLATED",
          "position": [
            103.77034978816715,
            1.4192107227680897,
            0
          ],
          "rotation": [
            0,
            1.1107430259071058,
            0
          ]
        },
        {
          "frame": 35,
          "state": "KEYFRAME",
          "position": [
            103.77033830303748,
            1.4192163084694085,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 36,
          "state": "INTERPOLATED",
          "position": [
            103.77032451985049,
            1.4192242342991785,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 37,
          "state": "INTERPOLATED",
          "position": [
            103.77031073666342,
            1.4192321601288662,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 38,
          "state": "INTERPOLATED",
          "position": [
            103.77029695347626,
            1.419240085958472,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 39,
          "state": "INTERPOLATED",
          "position": [
            103.770283170289,
            1.4192480117879955,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 40,
          "state": "INTERPOLATED",
          "position": [
            103.77026938710165,
            1.419255937617437,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 41,
          "state": "INTERPOLATED",
          "position": [
            103.7702556039142,
            1.4192638634467962,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 42,
          "state": "INTERPOLATED",
          "position": [
            103.77024182072665,
            1.4192717892760733,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 43,
          "state": "INTERPOLATED",
          "position": [
            103.77022803753901,
            1.4192797151052685,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 44,
          "state": "INTERPOLATED",
          "position": [
            103.77021425435129,
            1.4192876409343815,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 45,
          "state": "INTERPOLATED",
          "position": [
            103.77020047116346,
            1.4192955667634126,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 46,
          "state": "INTERPOLATED",
          "position": [
            103.77018668797554,
            1.4193034925923615,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 47,
          "state": "INTERPOLATED",
          "position": [
            103.77017290478751,
            1.4193114184212279,
            0
          ],
          "rotation": [
            0,
            1.0918826971801219,
            0
          ]
        },
        {
          "frame": 48,
          "state": "KEYFRAME",
          "position": [
            103.7701591215994,
            1.4193193442500127,
            0
          ],
          "rotation": [
            0,
            0.9408676673173344,
            0
          ]
        },
        {
          "frame": 49,
          "state": "INTERPOLATED",
          "position": [
            103.77014671684368,
            1.419328866522851,
            0
          ],
          "rotation": [
            0,
            0.9408676673173344,
            0
          ]
        },
        {
          "frame": 50,
          "state": "INTERPOLATED",
          "position": [
            103.77013431208785,
            1.4193383887956186,
            0
          ],
          "rotation": [
            0,
            0.9408676673173344,
            0
          ]
        },
        {
          "frame": 51,
          "state": "INTERPOLATED",
          "position": [
            103.77012190733191,
            1.4193479110683198,
            0
          ],
          "rotation": [
            0,
            0.9408676673173344,
            0
          ]
        },
        {
          "frame": 52,
          "state": "INTERPOLATED",
          "position": [
            103.77010950257588,
            1.4193574333409544,
            0
          ],
          "rotation": [
            0,
            0.9408676673173344,
            0
          ]
        },
        {
          "frame": 53,
          "state": "INTERPOLATED",
          "position": [
            103.77009709781974,
            1.4193669556135229,
            0
          ],
          "rotation": [
            0,
            0.9408676673173344,
            0
          ]
        },
        {
          "frame": 54,
          "state": "INTERPOLATED",
          "position": [
            103.77008469306348,
            1.4193764778860245,
            0
          ],
          "rotation": [
            0,
            0.9408676673173344,
            0
          ]
        },
        {
          "frame": 55,
          "state": "INTERPOLATED",
          "position": [
            103.77007228830715,
            1.4193860001584595,
            0
          ],
          "rotation": [
            0,
            0.9408676673173344,
            0
          ]
        },
        {
          "frame": 56,
          "state": "INTERPOLATED",
          "position": [
            103.7700598835507,
            1.4193955224308281,
            0
          ],
          "rotation": [
            0,
            0.9408676673173344,
            0
          ]
        },
        {
          "frame": 57,
          "state": "KEYFRAME",
          "position": [
            103.77004747879415,
            1.4194050447031303,
            0
          ],
          "rotation": [
            0,
            0.7752681501941547,
            0
          ]
        },
        {
          "frame": 58,
          "state": "INTERPOLATED",
          "position": [
            103.77003472710783,
            1.4194174452364339,
            0
          ],
          "rotation": [
            0,
            0.7752681501941547,
            0
          ]
        },
        {
          "frame": 59,
          "state": "INTERPOLATED",
          "position": [
            103.77002197542136,
            1.419429845769667,
            0
          ],
          "rotation": [
            0,
            0.7752681501941547,
            0
          ]
        },
        {
          "frame": 60,
          "state": "INTERPOLATED",
          "position": [
            103.77000922373475,
            1.4194422463028298,
            0
          ],
          "rotation": [
            0,
            0.7752681501941547,
            0
          ]
        },
        {
          "frame": 61,
          "state": "INTERPOLATED",
          "position": [
            103.76999647204802,
            1.4194546468359222,
            0
          ],
          "rotation": [
            0,
            0.7752681501941547,
            0
          ]
        },
        {
          "frame": 62,
          "state": "INTERPOLATED",
          "position": [
            103.76998372036114,
            1.4194670473689446,
            0
          ],
          "rotation": [
            0,
            0.7752681501941547,
            0
          ]
        },
        {
          "frame": 63,
          "state": "INTERPOLATED",
          "position": [
            103.76997096867413,
            1.4194794479018964,
            0
          ],
          "rotation": [
            0,
            0.7752681501941547,
            0
          ]
        },
        {
          "frame": 64,
          "state": "KEYFRAME",
          "position": [
            103.76995821698698,
            1.419491848434778,
            0
          ],
          "rotation": [
            0,
            0.6512807246976826,
            0
          ]
        },
        {
          "frame": 65,
          "state": "INTERPOLATED",
          "position": [
            103.76994748396082,
            1.4195061977677623,
            0
          ],
          "rotation": [
            0,
            0.6512807246976826,
            0
          ]
        },
        {
          "frame": 66,
          "state": "INTERPOLATED",
          "position": [
            103.76993675093453,
            1.4195205471007002,
            0
          ],
          "rotation": [
            0,
            0.6512807246976826,
            0
          ]
        },
        {
          "frame": 67,
          "state": "INTERPOLATED",
          "position": [
            103.7699260179081,
            1.4195348964335885,
            0
          ],
          "rotation": [
            0,
            0.6512807246976826,
            0
          ]
        },
        {
          "frame": 68,
          "state": "INTERPOLATED",
          "position": [
            103.76991528488156,
            1.4195492457664272,
            0
          ],
          "rotation": [
            0,
            0.6512807246976826,
            0
          ]
        },
        {
          "frame": 69,
          "state": "INTERPOLATED",
          "position": [
            103.76990455185488,
            1.4195635950992165,
            0
          ],
          "rotation": [
            0,
            0.6512807246976826,
            0
          ]
        },
        {
          "frame": 70,
          "state": "KEYFRAME",
          "position": [
            103.76989381882804,
            1.4195779444319554,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 71,
          "state": "INTERPOLATED",
          "position": [
            103.76988679857287,
            1.4195921476411306,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 72,
          "state": "INTERPOLATED",
          "position": [
            103.76987977831762,
            1.419606350850283,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 73,
          "state": "INTERPOLATED",
          "position": [
            103.76987275806228,
            1.4196205540594138,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 74,
          "state": "INTERPOLATED",
          "position": [
            103.76986573780685,
            1.419634757268523,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 75,
          "state": "INTERPOLATED",
          "position": [
            103.76985871755133,
            1.4196489604776115,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 76,
          "state": "INTERPOLATED",
          "position": [
            103.76985169729573,
            1.4196631636866786,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 77,
          "state": "INTERPOLATED",
          "position": [
            103.76984467704006,
            1.419677366895724,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 78,
          "state": "INTERPOLATED",
          "position": [
            103.76983765678429,
            1.4196915701047486,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 79,
          "state": "INTERPOLATED",
          "position": [
            103.76983063652843,
            1.4197057733137515,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 80,
          "state": "INTERPOLATED",
          "position": [
            103.76982361627248,
            1.4197199765227333,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 81,
          "state": "INTERPOLATED",
          "position": [
            103.76981659601645,
            1.4197341797316936,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 82,
          "state": "INTERPOLATED",
          "position": [
            103.76980957576033,
            1.4197483829406328,
            0
          ],
          "rotation": [
            0,
            0.4924562151911653,
            0
          ]
        },
        {
          "frame": 83,
          "state": "KEYFRAME",
          "position": [
            103.76980255550413,
            1.4197625861495504,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 84,
          "state": "INTERPOLATED",
          "position": [
            103.76980132090172,
            1.4197762226923822,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 85,
          "state": "INTERPOLATED",
          "position": [
            103.76980008629928,
            1.419789859235214,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 86,
          "state": "INTERPOLATED",
          "position": [
            103.76979885169683,
            1.4198034957780448,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 87,
          "state": "INTERPOLATED",
          "position": [
            103.76979761709437,
            1.419817132320875,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 88,
          "state": "INTERPOLATED",
          "position": [
            103.76979638249189,
            1.4198307688637044,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 89,
          "state": "INTERPOLATED",
          "position": [
            103.76979514788941,
            1.4198444054065338,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 90,
          "state": "INTERPOLATED",
          "position": [
            103.7697939132869,
            1.4198580419493618,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 91,
          "state": "INTERPOLATED",
          "position": [
            103.76979267868438,
            1.4198716784921959,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 92,
          "state": "INTERPOLATED",
          "position": [
            103.76979144408185,
            1.4198853150350164,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 93,
          "state": "INTERPOLATED",
          "position": [
            103.7697902094793,
            1.4198989515778426,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 94,
          "state": "INTERPOLATED",
          "position": [
            103.76978897487673,
            1.4199125881206682,
            0
          ],
          "rotation": [
            0,
            0.11846551678764027,
            0
          ]
        },
        {
          "frame": 95,
          "state": "KEYFRAME",
          "position": [
            103.76978774027415,
            1.4199262246634932,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 96,
          "state": "INTERPOLATED",
          "position": [
            103.76978803426515,
            1.419931821967855,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 97,
          "state": "INTERPOLATED",
          "position": [
            103.76978832825614,
            1.4199374192722174,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 98,
          "state": "INTERPOLATED",
          "position": [
            103.76978862224712,
            1.4199430165765794,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 99,
          "state": "INTERPOLATED",
          "position": [
            103.76978891623813,
            1.4199486138809414,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 100,
          "state": "INTERPOLATED",
          "position": [
            103.7697892102291,
            1.4199542111853036,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 101,
          "state": "INTERPOLATED",
          "position": [
            103.7697895042201,
            1.4199598084896656,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 102,
          "state": "INTERPOLATED",
          "position": [
            103.7697897982111,
            1.4199654057940279,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 103,
          "state": "INTERPOLATED",
          "position": [
            103.7697900922021,
            1.4199710030983894,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 104,
          "state": "INTERPOLATED",
          "position": [
            103.7697903861931,
            1.4199766004027514,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 105,
          "state": "INTERPOLATED",
          "position": [
            103.7697906801841,
            1.4199821977071132,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 106,
          "state": "INTERPOLATED",
          "position": [
            103.7697909741751,
            1.419987795011475,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 107,
          "state": "INTERPOLATED",
          "position": [
            103.7697912681661,
            1.4199933923158368,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 108,
          "state": "INTERPOLATED",
          "position": [
            103.76979156215711,
            1.4199989896201985,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 109,
          "state": "INTERPOLATED",
          "position": [
            103.76979185614812,
            1.4200045869245606,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 110,
          "state": "INTERPOLATED",
          "position": [
            103.76979215013911,
            1.4200101842289221,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 111,
          "state": "INTERPOLATED",
          "position": [
            103.76979244413012,
            1.4200157815332837,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 112,
          "state": "INTERPOLATED",
          "position": [
            103.76979273812114,
            1.420021378837645,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 113,
          "state": "INTERPOLATED",
          "position": [
            103.76979303211216,
            1.4200269761420066,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 114,
          "state": "INTERPOLATED",
          "position": [
            103.76979332610317,
            1.4200325734463681,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 115,
          "state": "INTERPOLATED",
          "position": [
            103.76979362009418,
            1.4200381707507295,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 116,
          "state": "INTERPOLATED",
          "position": [
            103.7697939140852,
            1.4200437680550912,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 117,
          "state": "INTERPOLATED",
          "position": [
            103.7697942080762,
            1.4200493653594526,
            0
          ],
          "rotation": [
            0,
            0.032381608427936935,
            0
          ]
        },
        {
          "frame": 118,
          "state": "KEYFRAME",
          "position": [
            103.76979450206723,
            1.420054962663814,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 119,
          "state": "INTERPOLATED",
          "position": [
            103.76979468185372,
            1.420055574450205,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 120,
          "state": "INTERPOLATED",
          "position": [
            103.76979486164022,
            1.4200561862365946,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 121,
          "state": "INTERPOLATED",
          "position": [
            103.7697950414267,
            1.4200567980229848,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 122,
          "state": "INTERPOLATED",
          "position": [
            103.76979522121322,
            1.4200574098093748,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 123,
          "state": "INTERPOLATED",
          "position": [
            103.7697954009997,
            1.4200580215957643,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 124,
          "state": "INTERPOLATED",
          "position": [
            103.76979558078621,
            1.4200586333821543,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 125,
          "state": "INTERPOLATED",
          "position": [
            103.76979576057269,
            1.420059245168544,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 126,
          "state": "INTERPOLATED",
          "position": [
            103.7697959403592,
            1.4200598569549339,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 127,
          "state": "INTERPOLATED",
          "position": [
            103.76979612014568,
            1.4200604687413236,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 128,
          "state": "INTERPOLATED",
          "position": [
            103.76979629993218,
            1.4200610805277132,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 129,
          "state": "INTERPOLATED",
          "position": [
            103.76979647971868,
            1.4200616923141032,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 130,
          "state": "INTERPOLATED",
          "position": [
            103.76979665950518,
            1.420062304100493,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 131,
          "state": "INTERPOLATED",
          "position": [
            103.76979683929167,
            1.4200629158868827,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 132,
          "state": "INTERPOLATED",
          "position": [
            103.76979701907817,
            1.4200635276732723,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 133,
          "state": "INTERPOLATED",
          "position": [
            103.76979719886467,
            1.4200641394596623,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 134,
          "state": "INTERPOLATED",
          "position": [
            103.76979737865116,
            1.4200647512460518,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 135,
          "state": "INTERPOLATED",
          "position": [
            103.76979755843765,
            1.4200653630324414,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 136,
          "state": "INTERPOLATED",
          "position": [
            103.76979773822416,
            1.4200659748188313,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 137,
          "state": "INTERPOLATED",
          "position": [
            103.76979791801065,
            1.4200665866052207,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 138,
          "state": "INTERPOLATED",
          "position": [
            103.76979809779715,
            1.4200671983916104,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 139,
          "state": "INTERPOLATED",
          "position": [
            103.76979827758365,
            1.4200678101780002,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 140,
          "state": "INTERPOLATED",
          "position": [
            103.76979845737013,
            1.4200684219643893,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 141,
          "state": "INTERPOLATED",
          "position": [
            103.76979863715664,
            1.4200690337507795,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 142,
          "state": "INTERPOLATED",
          "position": [
            103.76979881694312,
            1.4200696455371686,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 143,
          "state": "INTERPOLATED",
          "position": [
            103.76979899672963,
            1.4200702573235586,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 144,
          "state": "INTERPOLATED",
          "position": [
            103.76979917651613,
            1.420070869109948,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 145,
          "state": "INTERPOLATED",
          "position": [
            103.76979935630261,
            1.4200714808963375,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 146,
          "state": "INTERPOLATED",
          "position": [
            103.76979953608912,
            1.4200720926827268,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 147,
          "state": "INTERPOLATED",
          "position": [
            103.76979971587562,
            1.4200727044691164,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 148,
          "state": "INTERPOLATED",
          "position": [
            103.76979989566212,
            1.4200733162555061,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 149,
          "state": "INTERPOLATED",
          "position": [
            103.76980007544861,
            1.4200739280418955,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 150,
          "state": "INTERPOLATED",
          "position": [
            103.76980025523511,
            1.4200745398282848,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 151,
          "state": "INTERPOLATED",
          "position": [
            103.76980043502161,
            1.4200751516146743,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 152,
          "state": "INTERPOLATED",
          "position": [
            103.7698006148081,
            1.4200757634010637,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 153,
          "state": "INTERPOLATED",
          "position": [
            103.76980079459462,
            1.4200763751874532,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 154,
          "state": "INTERPOLATED",
          "position": [
            103.7698009743811,
            1.4200769869738425,
            0
          ],
          "rotation": [
            0,
            -0.17183655288477653,
            0
          ]
        },
        {
          "frame": 155,
          "state": "KEYFRAME",
          "position": [
            103.7698011541676,
            1.420077598760232,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 156,
          "state": "INTERPOLATED",
          "position": [
            103.7698016046746,
            1.420079542485391,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 157,
          "state": "INTERPOLATED",
          "position": [
            103.7698020551816,
            1.4200814862105506,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 158,
          "state": "INTERPOLATED",
          "position": [
            103.76980250568859,
            1.4200834299357106,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 159,
          "state": "INTERPOLATED",
          "position": [
            103.76980295619559,
            1.42008537366087,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 160,
          "state": "INTERPOLATED",
          "position": [
            103.76980340670258,
            1.4200873173860296,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 161,
          "state": "INTERPOLATED",
          "position": [
            103.76980385720958,
            1.420089261111189,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 162,
          "state": "INTERPOLATED",
          "position": [
            103.76980430771657,
            1.4200912048363485,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 163,
          "state": "INTERPOLATED",
          "position": [
            103.76980475822357,
            1.420093148561508,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 164,
          "state": "INTERPOLATED",
          "position": [
            103.76980520873057,
            1.420095092286667,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 165,
          "state": "INTERPOLATED",
          "position": [
            103.76980565923758,
            1.4200970360118261,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 166,
          "state": "INTERPOLATED",
          "position": [
            103.76980610974458,
            1.4200989797369852,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 167,
          "state": "INTERPOLATED",
          "position": [
            103.76980656025158,
            1.420100923462144,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 168,
          "state": "INTERPOLATED",
          "position": [
            103.7698070107586,
            1.420102867187303,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 169,
          "state": "INTERPOLATED",
          "position": [
            103.76980746126559,
            1.4201048109124619,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 170,
          "state": "INTERPOLATED",
          "position": [
            103.7698079117726,
            1.4201067546376203,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 171,
          "state": "INTERPOLATED",
          "position": [
            103.7698083622796,
            1.4201086983627789,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 172,
          "state": "INTERPOLATED",
          "position": [
            103.76980881278661,
            1.4201106420879372,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 173,
          "state": "INTERPOLATED",
          "position": [
            103.76980926329361,
            1.4201125858130956,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 174,
          "state": "INTERPOLATED",
          "position": [
            103.76980971380061,
            1.4201145295382542,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 175,
          "state": "INTERPOLATED",
          "position": [
            103.76981016430763,
            1.4201164732634124,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 176,
          "state": "INTERPOLATED",
          "position": [
            103.76981061481463,
            1.4201184169885708,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 177,
          "state": "INTERPOLATED",
          "position": [
            103.76981106532165,
            1.4201203607137287,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 178,
          "state": "INTERPOLATED",
          "position": [
            103.76981151582866,
            1.4201223044388867,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 179,
          "state": "INTERPOLATED",
          "position": [
            103.76981196633568,
            1.4201242481640446,
            0
          ],
          "rotation": [
            0,
            -0.19445210427481496,
            0
          ]
        },
        {
          "frame": 180,
          "state": "KEYFRAME",
          "position": [
            103.76981241684268,
            1.4201261918892023,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 181,
          "state": "INTERPOLATED",
          "position": [
            103.76981334832793,
            1.4201292961657874,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 182,
          "state": "INTERPOLATED",
          "position": [
            103.76981427981318,
            1.4201324004423737,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 183,
          "state": "INTERPOLATED",
          "position": [
            103.76981521129844,
            1.4201355047189594,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 184,
          "state": "INTERPOLATED",
          "position": [
            103.76981614278368,
            1.4201386089955446,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 185,
          "state": "INTERPOLATED",
          "position": [
            103.76981707426894,
            1.4201417132721295,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 186,
          "state": "INTERPOLATED",
          "position": [
            103.7698180057542,
            1.4201448175487141,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 187,
          "state": "INTERPOLATED",
          "position": [
            103.76981893723946,
            1.4201479218252981,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 188,
          "state": "INTERPOLATED",
          "position": [
            103.76981986872472,
            1.4201510261018817,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 189,
          "state": "INTERPOLATED",
          "position": [
            103.76982080021,
            1.420154130378465,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 190,
          "state": "INTERPOLATED",
          "position": [
            103.76982173169526,
            1.4201572346550482,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 191,
          "state": "INTERPOLATED",
          "position": [
            103.76982266318052,
            1.4201603389316306,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 192,
          "state": "INTERPOLATED",
          "position": [
            103.76982359466581,
            1.4201634432082129,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 193,
          "state": "INTERPOLATED",
          "position": [
            103.76982452615108,
            1.4201665474847949,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 194,
          "state": "INTERPOLATED",
          "position": [
            103.76982545763637,
            1.4201696517613764,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 195,
          "state": "INTERPOLATED",
          "position": [
            103.76982638912165,
            1.4201727560379573,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 196,
          "state": "INTERPOLATED",
          "position": [
            103.76982732060694,
            1.4201758603145382,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 197,
          "state": "INTERPOLATED",
          "position": [
            103.76982825209221,
            1.4201789645911185,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 198,
          "state": "INTERPOLATED",
          "position": [
            103.7698291835775,
            1.4201820688676985,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 199,
          "state": "INTERPOLATED",
          "position": [
            103.7698301150628,
            1.420185173144278,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 200,
          "state": "INTERPOLATED",
          "position": [
            103.76983104654809,
            1.4201882774208574,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 201,
          "state": "INTERPOLATED",
          "position": [
            103.7698319780334,
            1.420191381697436,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 202,
          "state": "INTERPOLATED",
          "position": [
            103.76983290951868,
            1.4201944859740148,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 203,
          "state": "INTERPOLATED",
          "position": [
            103.76983384100399,
            1.4201975902505928,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 204,
          "state": "INTERPOLATED",
          "position": [
            103.76983477248929,
            1.4202006945271706,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 205,
          "state": "INTERPOLATED",
          "position": [
            103.76983570397459,
            1.420203798803748,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 206,
          "state": "INTERPOLATED",
          "position": [
            103.7698366354599,
            1.420206903080325,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 207,
          "state": "INTERPOLATED",
          "position": [
            103.76983756694521,
            1.4202100073569017,
            0
          ],
          "rotation": [
            0,
            -0.2622006196971395,
            0
          ]
        },
        {
          "frame": 208,
          "state": "KEYFRAME",
          "position": [
            103.76983849843053,
            1.4202131116334777,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 209,
          "state": "INTERPOLATED",
          "position": [
            103.76984118825845,
            1.4202182617053165,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 210,
          "state": "INTERPOLATED",
          "position": [
            103.76984387808638,
            1.4202234117771488,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 211,
          "state": "INTERPOLATED",
          "position": [
            103.76984656791431,
            1.4202285618489778,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 212,
          "state": "INTERPOLATED",
          "position": [
            103.76984925774227,
            1.4202337119208042,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 213,
          "state": "INTERPOLATED",
          "position": [
            103.76985194757023,
            1.4202388619926265,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 214,
          "state": "INTERPOLATED",
          "position": [
            103.7698546373982,
            1.4202440120644466,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 215,
          "state": "INTERPOLATED",
          "position": [
            103.76985732722619,
            1.4202491621362632,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 216,
          "state": "INTERPOLATED",
          "position": [
            103.76986001705419,
            1.4202543122080769,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 217,
          "state": "INTERPOLATED",
          "position": [
            103.76986270688221,
            1.420259462279887,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 218,
          "state": "INTERPOLATED",
          "position": [
            103.76986539671023,
            1.4202646123516942,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 219,
          "state": "INTERPOLATED",
          "position": [
            103.76986808653825,
            1.4202697624234983,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 220,
          "state": "INTERPOLATED",
          "position": [
            103.76987077636632,
            1.4202749124952994,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 221,
          "state": "INTERPOLATED",
          "position": [
            103.76987346619437,
            1.420280062567097,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 222,
          "state": "INTERPOLATED",
          "position": [
            103.76987615602245,
            1.4202852126388918,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 223,
          "state": "INTERPOLATED",
          "position": [
            103.76987884585053,
            1.4202903627106835,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 224,
          "state": "INTERPOLATED",
          "position": [
            103.76988153567862,
            1.420295512782472,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 225,
          "state": "INTERPOLATED",
          "position": [
            103.76988422550673,
            1.4203006628542572,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 226,
          "state": "INTERPOLATED",
          "position": [
            103.76988691533485,
            1.4203058129260393,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 227,
          "state": "INTERPOLATED",
          "position": [
            103.76988960516299,
            1.4203109629978183,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 228,
          "state": "INTERPOLATED",
          "position": [
            103.76989229499112,
            1.4203161130695945,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 229,
          "state": "INTERPOLATED",
          "position": [
            103.76989498481927,
            1.4203212631413673,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 230,
          "state": "INTERPOLATED",
          "position": [
            103.76989767464745,
            1.420326413213137,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 231,
          "state": "INTERPOLATED",
          "position": [
            103.76990036447562,
            1.4203315632849036,
            0
          ],
          "rotation": [
            0,
            -0.4350573204184924,
            0
          ]
        },
        {
          "frame": 232,
          "state": "KEYFRAME",
          "position": [
            103.76990305430381,
            1.420336713356667,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 233,
          "state": "INTERPOLATED",
          "position": [
            103.76990786885735,
            1.420343258461476,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 234,
          "state": "INTERPOLATED",
          "position": [
            103.76991268341091,
            1.4203498035662767,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 235,
          "state": "INTERPOLATED",
          "position": [
            103.76991749796449,
            1.4203563486710673,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 236,
          "state": "INTERPOLATED",
          "position": [
            103.76992231251809,
            1.420362893775848,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 237,
          "state": "INTERPOLATED",
          "position": [
            103.76992712707172,
            1.4203694388806185,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 238,
          "state": "INTERPOLATED",
          "position": [
            103.76993194162539,
            1.4203759839853791,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 239,
          "state": "INTERPOLATED",
          "position": [
            103.76993675617909,
            1.42038252909013,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 240,
          "state": "INTERPOLATED",
          "position": [
            103.7699415707328,
            1.4203890741948704,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 241,
          "state": "INTERPOLATED",
          "position": [
            103.76994638528654,
            1.420395619299601,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 242,
          "state": "INTERPOLATED",
          "position": [
            103.76995119984032,
            1.4204021644043214,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 243,
          "state": "INTERPOLATED",
          "position": [
            103.76995601439413,
            1.420408709509032,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 244,
          "state": "INTERPOLATED",
          "position": [
            103.76996082894794,
            1.4204152546137323,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 245,
          "state": "INTERPOLATED",
          "position": [
            103.7699656435018,
            1.4204217997184228,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 246,
          "state": "INTERPOLATED",
          "position": [
            103.76997045805568,
            1.4204283448231032,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 247,
          "state": "INTERPOLATED",
          "position": [
            103.76997527260959,
            1.4204348899277737,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 248,
          "state": "INTERPOLATED",
          "position": [
            103.76998008716353,
            1.420441435032434,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 249,
          "state": "INTERPOLATED",
          "position": [
            103.76998490171749,
            1.4204479801370844,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 250,
          "state": "INTERPOLATED",
          "position": [
            103.7699897162715,
            1.4204545252417247,
            0
          ],
          "rotation": [
            0,
            -0.6023346596419921,
            0
          ]
        },
        {
          "frame": 251,
          "state": "KEYFRAME",
          "position": [
            103.7699945308255,
            1.420461070346355,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 252,
          "state": "INTERPOLATED",
          "position": [
            103.77000349960858,
            1.4204700902769074,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 253,
          "state": "INTERPOLATED",
          "position": [
            103.7700124683917,
            1.4204791102074241,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 254,
          "state": "INTERPOLATED",
          "position": [
            103.77002143717492,
            1.4204881301379062,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 255,
          "state": "INTERPOLATED",
          "position": [
            103.77003040595818,
            1.4204971500683532,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 256,
          "state": "INTERPOLATED",
          "position": [
            103.77003937474154,
            1.4205061699987656,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 257,
          "state": "INTERPOLATED",
          "position": [
            103.77004834352495,
            1.4205151899291433,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 258,
          "state": "INTERPOLATED",
          "position": [
            103.77005731230844,
            1.420524209859486,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 259,
          "state": "INTERPOLATED",
          "position": [
            103.770066281092,
            1.420533229789794,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 260,
          "state": "INTERPOLATED",
          "position": [
            103.77007524987562,
            1.4205422497200673,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 261,
          "state": "INTERPOLATED",
          "position": [
            103.77008421865932,
            1.4205512696503053,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 262,
          "state": "INTERPOLATED",
          "position": [
            103.77009318744308,
            1.420560289580509,
            0
          ],
          "rotation": [
            0,
            -0.7339379580449379,
            0
          ]
        },
        {
          "frame": 263,
          "state": "KEYFRAME",
          "position": [
            103.77010215622691,
            1.4205693095106777,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 264,
          "state": "INTERPOLATED",
          "position": [
            103.77011213450686,
            1.420577451449717,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 265,
          "state": "INTERPOLATED",
          "position": [
            103.77012211278688,
            1.4205855933887148,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 266,
          "state": "INTERPOLATED",
          "position": [
            103.77013209106697,
            1.4205937353276696,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 267,
          "state": "INTERPOLATED",
          "position": [
            103.77014206934713,
            1.4206018772665814,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 268,
          "state": "INTERPOLATED",
          "position": [
            103.77015204762736,
            1.42061001920545,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 269,
          "state": "INTERPOLATED",
          "position": [
            103.77016202590767,
            1.4206181611442756,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 270,
          "state": "INTERPOLATED",
          "position": [
            103.77017200418804,
            1.4206263030830582,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 271,
          "state": "INTERPOLATED",
          "position": [
            103.77018198246849,
            1.420634445021798,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 272,
          "state": "INTERPOLATED",
          "position": [
            103.77019196074899,
            1.4206425869604942,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 273,
          "state": "INTERPOLATED",
          "position": [
            103.77020193902958,
            1.4206507288991477,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 274,
          "state": "INTERPOLATED",
          "position": [
            103.77021191731023,
            1.420658870837758,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 275,
          "state": "INTERPOLATED",
          "position": [
            103.77022189559095,
            1.420667012776325,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 276,
          "state": "INTERPOLATED",
          "position": [
            103.77023187387174,
            1.4206751547148493,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 277,
          "state": "INTERPOLATED",
          "position": [
            103.77024185215261,
            1.4206832966533303,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 278,
          "state": "INTERPOLATED",
          "position": [
            103.77025183043354,
            1.4206914385917686,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 279,
          "state": "INTERPOLATED",
          "position": [
            103.77026180871455,
            1.4206995805301634,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 280,
          "state": "INTERPOLATED",
          "position": [
            103.77027178699562,
            1.4207077224685152,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 281,
          "state": "INTERPOLATED",
          "position": [
            103.77028176527676,
            1.4207158644068238,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 282,
          "state": "INTERPOLATED",
          "position": [
            103.77029174355799,
            1.4207240063450899,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 283,
          "state": "INTERPOLATED",
          "position": [
            103.77030172183926,
            1.4207321482833124,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 284,
          "state": "INTERPOLATED",
          "position": [
            103.77031170012062,
            1.4207402902214916,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 285,
          "state": "INTERPOLATED",
          "position": [
            103.77032167840206,
            1.4207484321596286,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 286,
          "state": "INTERPOLATED",
          "position": [
            103.77033165668355,
            1.4207565740977217,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 287,
          "state": "INTERPOLATED",
          "position": [
            103.77034163496512,
            1.420764716035772,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 288,
          "state": "INTERPOLATED",
          "position": [
            103.77035161324675,
            1.4207728579737795,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 289,
          "state": "INTERPOLATED",
          "position": [
            103.77036159152847,
            1.4207809999117438,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 290,
          "state": "INTERPOLATED",
          "position": [
            103.77037156981024,
            1.420789141849665,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 291,
          "state": "INTERPOLATED",
          "position": [
            103.77038154809209,
            1.4207972837875429,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 292,
          "state": "INTERPOLATED",
          "position": [
            103.770391526374,
            1.420805425725378,
            0
          ],
          "rotation": [
            0,
            -0.9059573251272349,
            0
          ]
        },
        {
          "frame": 293,
          "state": "KEYFRAME",
          "position": [
            103.770401504656,
            1.4208135676631697,
            0
          ],
          "rotation": [
            0,
            -0.9306007568200447,
            0
          ]
        }
      ]
    },
    {
      "type": "MOTORCYCLE",
      "involvement": "NONE",
      "severity": 0,
      "timestamps": {
        "T1": -1,
        "T2": -1,
        "T3": -1,
        "T4": -1,
        "T5": -1
      },
      "scale": [
        1,
        1,
        1
      ],
      "trajectory": [
        {
          "frame": 179,
          "state": "KEYFRAME",
          "position": [
            103.76979869281246,
            1.4197500183766008,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 180,
          "state": "INTERPOLATED",
          "position": [
            103.76979738048996,
            1.4197642366343715,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 181,
          "state": "INTERPOLATED",
          "position": [
            103.76979606816747,
            1.4197784548921464,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 182,
          "state": "INTERPOLATED",
          "position": [
            103.76979475584494,
            1.419792673149918,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 183,
          "state": "INTERPOLATED",
          "position": [
            103.76979344352242,
            1.4198068914076885,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 184,
          "state": "INTERPOLATED",
          "position": [
            103.76979213119985,
            1.4198211096654583,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 185,
          "state": "INTERPOLATED",
          "position": [
            103.7697908188773,
            1.419835327923227,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 186,
          "state": "INTERPOLATED",
          "position": [
            103.7697895065547,
            1.4198495461809957,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 187,
          "state": "INTERPOLATED",
          "position": [
            103.76978819423209,
            1.4198637644387633,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 188,
          "state": "INTERPOLATED",
          "position": [
            103.76978688190948,
            1.4198779826965302,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 189,
          "state": "INTERPOLATED",
          "position": [
            103.76978556958684,
            1.4198922009542962,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 190,
          "state": "INTERPOLATED",
          "position": [
            103.76978425726419,
            1.4199064192120616,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 191,
          "state": "INTERPOLATED",
          "position": [
            103.76978294494154,
            1.419920637469826,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 192,
          "state": "INTERPOLATED",
          "position": [
            103.76978163261884,
            1.41993485572759,
            0
          ],
          "rotation": [
            0,
            0.22372686622674534,
            0
          ]
        },
        {
          "frame": 193,
          "state": "KEYFRAME",
          "position": [
            103.76978032029615,
            1.419949073985353,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 194,
          "state": "INTERPOLATED",
          "position": [
            103.76978212981975,
            1.4199609420877848,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 195,
          "state": "INTERPOLATED",
          "position": [
            103.76978393934337,
            1.4199728101902143,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 196,
          "state": "INTERPOLATED",
          "position": [
            103.76978574886702,
            1.4199846782926422,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 197,
          "state": "INTERPOLATED",
          "position": [
            103.76978755839066,
            1.4199965463950686,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 198,
          "state": "INTERPOLATED",
          "position": [
            103.76978936791436,
            1.4200084144974938,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 199,
          "state": "INTERPOLATED",
          "position": [
            103.76979117743804,
            1.4200202825999173,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 200,
          "state": "INTERPOLATED",
          "position": [
            103.76979298696175,
            1.4200321507023395,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 201,
          "state": "INTERPOLATED",
          "position": [
            103.76979479648548,
            1.4200440188047603,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 202,
          "state": "INTERPOLATED",
          "position": [
            103.76979660600924,
            1.4200558869071795,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 203,
          "state": "INTERPOLATED",
          "position": [
            103.76979841553302,
            1.4200677550095977,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 204,
          "state": "INTERPOLATED",
          "position": [
            103.76980022505681,
            1.4200796231120143,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 205,
          "state": "INTERPOLATED",
          "position": [
            103.76980203458061,
            1.4200914912144293,
            0
          ],
          "rotation": [
            0,
            -0.0690602165534668,
            0
          ]
        },
        {
          "frame": 206,
          "state": "KEYFRAME",
          "position": [
            103.76980384410443,
            1.420103359316843,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 207,
          "state": "INTERPOLATED",
          "position": [
            103.76980500154752,
            1.4201080748355803,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 208,
          "state": "INTERPOLATED",
          "position": [
            103.7698061589906,
            1.4201127903543191,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 209,
          "state": "INTERPOLATED",
          "position": [
            103.7698073164337,
            1.4201175058730573,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 210,
          "state": "INTERPOLATED",
          "position": [
            103.7698084738768,
            1.4201222213917948,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 211,
          "state": "INTERPOLATED",
          "position": [
            103.7698096313199,
            1.4201269369105318,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 212,
          "state": "INTERPOLATED",
          "position": [
            103.76981078876301,
            1.4201316524292682,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 213,
          "state": "INTERPOLATED",
          "position": [
            103.76981194620612,
            1.4201363679480041,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 214,
          "state": "INTERPOLATED",
          "position": [
            103.76981310364924,
            1.4201410834667392,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 215,
          "state": "INTERPOLATED",
          "position": [
            103.76981426109236,
            1.4201457989854736,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 216,
          "state": "INTERPOLATED",
          "position": [
            103.76981541853549,
            1.4201505145042075,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 217,
          "state": "INTERPOLATED",
          "position": [
            103.76981657597862,
            1.420155230022941,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 218,
          "state": "INTERPOLATED",
          "position": [
            103.76981773342176,
            1.4201599455416738,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 219,
          "state": "INTERPOLATED",
          "position": [
            103.7698188908649,
            1.4201646610604062,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 220,
          "state": "INTERPOLATED",
          "position": [
            103.76982004830805,
            1.4201693765791377,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 221,
          "state": "INTERPOLATED",
          "position": [
            103.76982120575119,
            1.4201740920978692,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 222,
          "state": "INTERPOLATED",
          "position": [
            103.76982236319435,
            1.4201788076165995,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 223,
          "state": "INTERPOLATED",
          "position": [
            103.76982352063752,
            1.4201835231353297,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 224,
          "state": "INTERPOLATED",
          "position": [
            103.76982467808067,
            1.420188238654059,
            0
          ],
          "rotation": [
            0,
            -0.2993257234741909,
            0
          ]
        },
        {
          "frame": 225,
          "state": "KEYFRAME",
          "position": [
            103.76982583552385,
            1.4201929541727876,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 226,
          "state": "INTERPOLATED",
          "position": [
            103.76982674537273,
            1.4201961630713567,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 227,
          "state": "INTERPOLATED",
          "position": [
            103.76982765522163,
            1.4201993719699246,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 228,
          "state": "INTERPOLATED",
          "position": [
            103.76982856507053,
            1.4202025808684922,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 229,
          "state": "INTERPOLATED",
          "position": [
            103.76982947491943,
            1.42020578976706,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 230,
          "state": "INTERPOLATED",
          "position": [
            103.76983038476833,
            1.4202089986656268,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 231,
          "state": "INTERPOLATED",
          "position": [
            103.76983129461722,
            1.4202122075641936,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 232,
          "state": "INTERPOLATED",
          "position": [
            103.76983220446614,
            1.42021541646276,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 233,
          "state": "INTERPOLATED",
          "position": [
            103.76983311431503,
            1.420218625361326,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 234,
          "state": "INTERPOLATED",
          "position": [
            103.76983402416394,
            1.4202218342598916,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 235,
          "state": "INTERPOLATED",
          "position": [
            103.76983493401286,
            1.420225043158457,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 236,
          "state": "INTERPOLATED",
          "position": [
            103.76983584386178,
            1.420228252057022,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 237,
          "state": "INTERPOLATED",
          "position": [
            103.76983675371069,
            1.4202314609555866,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 238,
          "state": "INTERPOLATED",
          "position": [
            103.76983766355961,
            1.4202346698541508,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 239,
          "state": "INTERPOLATED",
          "position": [
            103.76983857340855,
            1.4202378787527143,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 240,
          "state": "INTERPOLATED",
          "position": [
            103.76983948325747,
            1.420241087651278,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 241,
          "state": "INTERPOLATED",
          "position": [
            103.76984039310639,
            1.4202442965498414,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 242,
          "state": "INTERPOLATED",
          "position": [
            103.76984130295533,
            1.420247505448404,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 243,
          "state": "INTERPOLATED",
          "position": [
            103.76984221280426,
            1.4202507143469667,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 244,
          "state": "INTERPOLATED",
          "position": [
            103.7698431226532,
            1.4202539232455285,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 245,
          "state": "INTERPOLATED",
          "position": [
            103.76984403250214,
            1.4202571321440902,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 246,
          "state": "INTERPOLATED",
          "position": [
            103.76984494235109,
            1.4202603410426515,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 247,
          "state": "INTERPOLATED",
          "position": [
            103.76984585220004,
            1.4202635499412126,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 248,
          "state": "INTERPOLATED",
          "position": [
            103.76984676204897,
            1.4202667588397733,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 249,
          "state": "INTERPOLATED",
          "position": [
            103.76984767189792,
            1.4202699677383335,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 250,
          "state": "INTERPOLATED",
          "position": [
            103.76984858174687,
            1.4202731766368932,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 251,
          "state": "INTERPOLATED",
          "position": [
            103.76984949159583,
            1.420276385535453,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 252,
          "state": "INTERPOLATED",
          "position": [
            103.76985040144477,
            1.420279594434012,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 253,
          "state": "INTERPOLATED",
          "position": [
            103.76985131129373,
            1.4202828033325712,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 254,
          "state": "INTERPOLATED",
          "position": [
            103.7698522211427,
            1.4202860122311294,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 255,
          "state": "INTERPOLATED",
          "position": [
            103.76985313099166,
            1.4202892211296878,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 256,
          "state": "INTERPOLATED",
          "position": [
            103.76985404084063,
            1.4202924300282456,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 257,
          "state": "INTERPOLATED",
          "position": [
            103.7698549506896,
            1.4202956389268029,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 258,
          "state": "INTERPOLATED",
          "position": [
            103.76985586053857,
            1.4202988478253598,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 259,
          "state": "INTERPOLATED",
          "position": [
            103.76985677038755,
            1.4203020567239166,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 260,
          "state": "INTERPOLATED",
          "position": [
            103.76985768023651,
            1.420305265622473,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 261,
          "state": "INTERPOLATED",
          "position": [
            103.7698585900855,
            1.4203084745210288,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 262,
          "state": "INTERPOLATED",
          "position": [
            103.76985949993448,
            1.4203116834195846,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 263,
          "state": "INTERPOLATED",
          "position": [
            103.76986040978346,
            1.4203148923181401,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 264,
          "state": "INTERPOLATED",
          "position": [
            103.76986131963245,
            1.4203181012166948,
            0
          ],
          "rotation": [
            0,
            -0.20684620623327712,
            0
          ]
        },
        {
          "frame": 265,
          "state": "KEYFRAME",
          "position": [
            103.76986222948145,
            1.4203213101152492,
            0
          ],
          "rotation": [
            0,
            -0.4076110306027085,
            0
          ]
        },
        {
          "frame": 266,
          "state": "INTERPOLATED",
          "position": [
            103.76986218260032,
            1.4203224192885213,
            0
          ],
          "rotation": [
            0,
            -0.4076110306027085,
            0
          ]
        },
        {
          "frame": 267,
          "state": "INTERPOLATED",
          "position": [
            103.76986213571921,
            1.420323528461793,
            0
          ],
          "rotation": [
            0,
            -0.4076110306027085,
            0
          ]
        },
        {
          "frame": 268,
          "state": "INTERPOLATED",
          "position": [
            103.7698620888381,
            1.4203246376350651,
            0
          ],
          "rotation": [
            0,
            -0.4076110306027085,
            0
          ]
        },
        {
          "frame": 269,
          "state": "INTERPOLATED",
          "position": [
            103.76986204195698,
            1.4203257468083372,
            0
          ],
          "rotation": [
            0,
            -0.4076110306027085,
            0
          ]
        },
        {
          "frame": 270,
          "state": "INTERPOLATED",
          "position": [
            103.76986199507586,
            1.4203268559816093,
            0
          ],
          "rotation": [
            0,
            -0.4076110306027085,
            0
          ]
        },
        {
          "frame": 271,
          "state": "INTERPOLATED",
          "position": [
            103.76986194819474,
            1.4203279651548815,
            0
          ],
          "rotation": [
            0,
            -0.4076110306027085,
            0
          ]
        },
        {
          "frame": 272,
          "state": "INTERPOLATED",
          "position": [
            103.76986190131363,
            1.4203290743281536,
            0
          ],
          "rotation": [
            0,
            -0.4076110306027085,
            0
          ]
        },
        {
          "frame": 273,
          "state": "INTERPOLATED",
          "position": [
            103.7698618544325,
            1.4203301835014255,
            0
          ],
          "rotation": [
            0,
            -0.4076110306027085,
            0
          ]
        },
        {
          "frame": 274,
          "state": "INTERPOLATED",
          "position": [
            103.7698618075514,
            1.4203312926746974,
            0
          ],
          "rotation": [
            0,
            -0.4076110306027085,
            0
          ]
        },
        {
          "frame": 275,
          "state": "INTERPOLATED",
          "position": [
            103.76986176067028,
            1.4203324018479695,
            0
          ],
          "rotation": [
            0,
            -0.4076110306027085,
            0
          ]
        },
        {
          "frame": 276,
          "state": "KEYFRAME",
          "position": [
            103.76986171378915,
            1.4203335110212416,
            0
          ],
          "rotation": [
            0,
            -0.29172316019758476,
            0
          ]
        }
      ]
    }
  ]
}
s=json.dumps(accident)
print(s)