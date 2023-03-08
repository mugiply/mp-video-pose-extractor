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
  // bodyVector
  v: number[][];
  // handVector
  h?: (number[] | null)[];
  // extendedData
  e?: { [key: string]: any };
  // mergedTimeMiliseconds
  mt?: number;
  // mergedDurationMiliseconds
  md?: number;
}
