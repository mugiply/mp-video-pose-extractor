export interface PoseVector {
  // 右腕
  rightWristToRightElbow: number[];
  rightElbowToRightShoulder: number[];
  // 左腕
  leftWristToLeftElbow: number[];
  leftElbowToLeftShoulder: number[];
  // 右親指
  rightThumbToWrist: number[];
  // 左親指
  leftThumbToWrist: number[];
  // 右人差し指
  rightIndexFingerToWrist: number[];
  // 左人差し指
  leftIndexFingerToWrist: number[];
  // 右小指
  rightPinkyFingerToWrist: number[];
  // 左小指
  leftPinkyFingerToWrist: number[];
}
