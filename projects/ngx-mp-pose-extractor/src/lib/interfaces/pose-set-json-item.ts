export interface PoseSetJsonItem {
  // timeMiliseconds
  t: number;
  // durationMiliseconds
  d: number;
  // pose
  p?: number[][];
  // leftHand
  l?: number[][];
  // rightHand
  r?: number[][];
  // body-vectors
  v: number[][];
  // hand-vectors
  h?: number[][];
}
