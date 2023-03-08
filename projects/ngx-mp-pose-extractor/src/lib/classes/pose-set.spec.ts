import { PoseSetJsonItem } from '../interfaces/pose-set-json-item';
import { PoseSet } from './pose-set';

describe('PoseSet', () => {
  it('初期化', () => {
    expect(new PoseSet()).toBeTruthy();
  });

  it('JSON形式のポーズセットの読み込み', () => {
    const poseSet = new PoseSet();
    const examplePoseSetJson = require('../../../test-assets/pose-set.json');
    poseSet.loadJson(examplePoseSetJson);

    expect(poseSet.getNumberOfPoses()).toBe(57);
    expect(poseSet.getPoses()[0].timeMiliseconds).toBe(712);
  });

  it('ポーズの追加', async () => {
    const poseSet = new PoseSet();
    poseSet.setVideoName('test');
    poseSet.setVideoMetaData(1080, 2040, 1000);

    const examplePoseSetJson = require('../../../test-assets/pose-set.json');
    const poseSetJsonItems: PoseSetJsonItem[] = examplePoseSetJson.poses;

    for (let i = 0; i < 30; i++) {
      const poseSetJsonItem = poseSetJsonItems[i];
      const results: any = {
        poseLandmarks: [],
        ea: poseSetJsonItem.p.map((v) => {
          return {
            x: v[0],
            y: v[1],
            z: v[2],
            visibility: v[3],
          };
        }),
        leftHandLandmarks: poseSetJsonItem.l
          ? poseSetJsonItem.l.map((v) => {
              return {
                x: v[0],
                y: v[1],
                z: v[2],
              };
            })
          : undefined,
        rightHandLandmarks: poseSetJsonItem.r
          ? poseSetJsonItem.r.map((v) => {
              return {
                x: v[0],
                y: v[1],
                z: v[2],
              };
            })
          : undefined,
      };
      poseSet.pushPose(poseSetJsonItem.t, null, null, null, results);
    }

    // ファイナライズ前
    expect(poseSet.getNumberOfPoses()).toBe(27);
    expect(poseSet.getPoseByTime(1144).debug.duplicatedItems).toEqual([
      {
        timeMiliseconds: 712,
        durationMiliseconds: 211,
      },
      {
        timeMiliseconds: 923,
        durationMiliseconds: 221,
      },
    ]);
    expect(poseSet.getPoseByTime(7849)).toBeDefined();

    // ファイナライズ後
    poseSet.finalize(true);
    expect(poseSet.getNumberOfPoses()).toBe(25);
    expect(poseSet.getPoseByTime(1568).debug.duplicatedItems.length).toBe(1);
    expect(poseSet.getPoseByTime(7849)).toBeDefined();
  });
});
