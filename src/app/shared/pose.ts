export interface PoseItem {
  t: number;
  pose: number[][];
  vectors: {
    rightWristToRightElbow: number[];
    rightElbowToRightShoulder: number[];
    leftWristToLeftElbow: number[];
    leftElbowToLeftShoulder: number[];
  };
  frameImageDataUrl?: string;
}

export interface PoseJson {
  generator: string;
  version: number;
  video: {
    width: number;
    height: number;
    duration: number;
  };
  poses: PoseItem[];
  poseLandmarkMapppings: string[];
}
