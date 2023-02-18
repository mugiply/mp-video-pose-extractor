export interface HandVector {
  // 右手 - 親指
  rightThumbTipToFirstJoint: number[];
  rightThumbFirstJointToSecondJoint: number[];
  // 右手 - 人差し指
  rightIndexFingerTipToFirstJoint: number[];
  rightIndexFingerFirstJointToSecondJoint: number[];
  // 右手 - 中指
  rightMiddleFingerTipToFirstJoint: number[];
  rightMiddleFingerFirstJointToSecondJoint: number[];
  // 右手 - 薬指
  rightRingFingerTipToFirstJoint: number[];
  rightRingFingerFirstJointToSecondJoint: number[];
  // 右手 - 小指
  rightPinkyFingerTipToFirstJoint: number[];
  rightPinkyFingerFirstJointToSecondJoint: number[];
  // 左手 - 親指
  leftThumbTipToFirstJoint: number[];
  leftThumbFirstJointToSecondJoint: number[];
  // 左手 - 人差し指
  leftIndexFingerTipToFirstJoint: number[];
  leftIndexFingerFirstJointToSecondJoint: number[];
  // 左手 - 中指
  leftMiddleFingerTipToFirstJoint: number[];
  leftMiddleFingerFirstJointToSecondJoint: number[];
  // 左手 - 薬指
  leftRingFingerTipToFirstJoint: number[];
  leftRingFingerFirstJointToSecondJoint: number[];
  // 左手 - 小指
  leftPinkyFingerTipToFirstJoint: number[];
  leftPinkyFingerFirstJointToSecondJoint: number[];
}
