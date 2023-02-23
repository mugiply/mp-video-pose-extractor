import { POSE_LANDMARKS, Results } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
import { PoseSetItem } from '../interfaces/pose-set-item';
import { PoseSetJson } from '../interfaces/pose-set-json';
import { PoseSetJsonItem } from '../interfaces/pose-set-json-item';
import { BodyVector } from '../interfaces/body-vector';

// @ts-ignore
import cosSimilarity from 'cos-similarity';
import { SimilarPoseItem } from '../interfaces/similar-pose-item';
import { ImageTrimmer } from './internals/image-trimmer';
import { HandVector } from '../interfaces/hand-vector';

export class PoseSet {
  public generator?: string;
  public version?: number;
  private videoMetadata!: {
    name: string;
    width: number;
    height: number;
    duration: number;
    firstPoseDetectedTime: number;
  };
  public poses: PoseSetItem[] = [];
  public isFinalized?: boolean = false;

  // BodyVector のキー名
  public static readonly BODY_VECTOR_MAPPINGS = [
    // 右腕
    'rightWristToRightElbow',
    'rightElbowToRightShoulder',
    // 左腕
    'leftWristToLeftElbow',
    'leftElbowToLeftShoulder',
  ];

  // HandVector のキー名
  public static readonly HAND_VECTOR_MAPPINGS = [
    // 右手 - 親指
    'rightThumbTipToFirstJoint',
    'rightThumbFirstJointToSecondJoint',
    // 右手 - 人差し指
    'rightIndexFingerTipToFirstJoint',
    'rightIndexFingerFirstJointToSecondJoint',
    // 右手 - 中指
    'rightMiddleFingerTipToFirstJoint',
    'rightMiddleFingerFirstJointToSecondJoint',
    // 右手 - 薬指
    'rightRingFingerTipToFirstJoint',
    'rightRingFingerFirstJointToSecondJoint',
    // 右手 - 小指
    'rightPinkyFingerTipToFirstJoint',
    'rightPinkyFingerFirstJointToSecondJoint',
    // 左手 - 親指
    'leftThumbTipToFirstJoint',
    'leftThumbFirstJointToSecondJoint',
    // 左手 - 人差し指
    'leftIndexFingerTipToFirstJoint',
    'leftIndexFingerFirstJointToSecondJoint',
    // 左手 - 中指
    'leftMiddleFingerTipToFirstJoint',
    'leftMiddleFingerFirstJointToSecondJoint',
    // 左手 - 薬指
    'leftRingFingerTipToFirstJoint',
    'leftRingFingerFirstJointToSecondJoint',
    // 左手 - 小指
    'leftPinkyFingerTipToFirstJoint',
    'leftPinkyFingerFirstJointToSecondJoint',
    // 右足
    'rightAnkleToRightKnee',
    'rightKneeToRightHip',
    // 左足
    'leftAnkleToLeftKnee',
    'leftKneeToLeftHip',
    // 胴体
    'rightHipToLeftHip',
    'rightShoulderToLeftShoulder',
  ];

  // 画像書き出し時の設定
  private readonly IMAGE_WIDTH: number = 1080;
  private readonly IMAGE_MIME: 'image/jpeg' | 'image/png' | 'image/webp' =
    'image/webp';
  private readonly IMAGE_QUALITY = 0.8;

  // 画像の余白除去
  private readonly IMAGE_MARGIN_TRIMMING_COLOR = '#000000';
  private readonly IMAGE_MARGIN_TRIMMING_DIFF_THRESHOLD = 50;

  // 画像の背景色置換
  private readonly IMAGE_BACKGROUND_REPLACE_SRC_COLOR = '#016AFD';
  private readonly IMAGE_BACKGROUND_REPLACE_DST_COLOR = '#FFFFFF00';
  private readonly IMAGE_BACKGROUND_REPLACE_DIFF_THRESHOLD = 130;

  constructor() {
    this.videoMetadata = {
      name: '',
      width: 0,
      height: 0,
      duration: 0,
      firstPoseDetectedTime: 0,
    };
  }

  getVideoName() {
    return this.videoMetadata.name;
  }

  setVideoName(videoName: string) {
    this.videoMetadata.name = videoName;
  }

  setVideoMetaData(width: number, height: number, duration: number) {
    this.videoMetadata.width = width;
    this.videoMetadata.height = height;
    this.videoMetadata.duration = duration;
  }

  getNumberOfPoses(): number {
    if (this.poses === undefined) return -1;
    return this.poses.length;
  }

  getPoses(): PoseSetItem[] {
    if (this.poses === undefined) return [];
    return this.poses;
  }

  getPoseByTime(timeMiliseconds: number): PoseSetItem | undefined {
    if (this.poses === undefined) return undefined;
    return this.poses.find((pose) => pose.timeMiliseconds === timeMiliseconds);
  }

  pushPose(
    videoTimeMiliseconds: number,
    frameImageDataUrl: string | undefined,
    poseImageDataUrl: string | undefined,
    faceFrameImageDataUrl: string | undefined,
    results: Results
  ): PoseSetItem | undefined {
    if (results.poseLandmarks === undefined) return;

    if (this.poses.length === 0) {
      this.videoMetadata.firstPoseDetectedTime = videoTimeMiliseconds;
    }

    const poseLandmarksWithWorldCoordinate: any[] = (results as any).ea
      ? (results as any).ea
      : [];
    if (poseLandmarksWithWorldCoordinate.length === 0) {
      console.warn(
        `[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the pose with the world coordinate`,
        results
      );
      return;
    }

    const bodyVector = PoseSet.getBodyVector(poseLandmarksWithWorldCoordinate);
    if (!bodyVector) {
      console.warn(
        `[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the body vector`,
        poseLandmarksWithWorldCoordinate
      );
      return;
    }

    if (
      results.leftHandLandmarks === undefined &&
      results.rightHandLandmarks === undefined
    ) {
      console.warn(
        `[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the hand landmarks`,
        results
      );
    } else if (results.leftHandLandmarks === undefined) {
      console.warn(
        `[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the left hand landmarks`,
        results
      );
    } else if (results.rightHandLandmarks === undefined) {
      console.warn(
        `[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the right hand landmarks`,
        results
      );
    }

    const handVector = PoseSet.getHandVectors(
      results.leftHandLandmarks,
      results.rightHandLandmarks
    );
    if (!handVector) {
      console.warn(
        `[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the hand vector`,
        results
      );
    }

    const pose: PoseSetItem = {
      timeMiliseconds: videoTimeMiliseconds,
      durationMiliseconds: -1,
      pose: poseLandmarksWithWorldCoordinate.map((worldCoordinateLandmark) => {
        return [
          worldCoordinateLandmark.x,
          worldCoordinateLandmark.y,
          worldCoordinateLandmark.z,
          worldCoordinateLandmark.visibility,
        ];
      }),
      leftHand: results.leftHandLandmarks?.map((normalizedLandmark) => {
        return [
          normalizedLandmark.x,
          normalizedLandmark.y,
          normalizedLandmark.z,
        ];
      }),
      rightHand: results.leftHandLandmarks?.map((normalizedLandmark) => {
        return [
          normalizedLandmark.x,
          normalizedLandmark.y,
          normalizedLandmark.z,
        ];
      }),
      bodyVectors: bodyVector,
      handVectors: handVector,
      frameImageDataUrl: frameImageDataUrl,
      poseImageDataUrl: poseImageDataUrl,
      faceFrameImageDataUrl: faceFrameImageDataUrl,
      extendedData: {},
    };

    if (1 <= this.poses.length) {
      // 前回のポーズとの類似性をチェック
      const lastPose = this.poses[this.poses.length - 1];

      const isSimilarBodyPose = PoseSet.isSimilarBodyPose(
        lastPose.bodyVectors,
        pose.bodyVectors
      );

      let isSimilarHandPose = true;
      if (lastPose.handVectors && pose.handVectors) {
        isSimilarHandPose = PoseSet.isSimilarHandPose(
          lastPose.handVectors,
          pose.handVectors
        );
      } else if (!lastPose.handVectors && pose.handVectors) {
        isSimilarHandPose = false;
      }

      if (isSimilarBodyPose && isSimilarHandPose) {
        // 身体・手ともに類似ポーズならばスキップ
        return;
      }

      // 前回のポーズの持続時間を設定
      const poseDurationMiliseconds =
        videoTimeMiliseconds - lastPose.timeMiliseconds;
      this.poses[this.poses.length - 1].durationMiliseconds =
        poseDurationMiliseconds;
    }

    this.poses.push(pose);

    return pose;
  }

  async finalize() {
    if (0 == this.poses.length) {
      this.isFinalized = true;
      return;
    }

    // 最後のポーズの持続時間を設定
    if (1 <= this.poses.length) {
      const lastPose = this.poses[this.poses.length - 1];
      if (lastPose.durationMiliseconds == -1) {
        const poseDurationMiliseconds =
          this.videoMetadata.duration - lastPose.timeMiliseconds;
        this.poses[this.poses.length - 1].durationMiliseconds =
          poseDurationMiliseconds;
      }
    }

    // 重複ポーズを除去
    this.removeDuplicatedPoses();

    // 最初のポーズを除去
    this.poses.shift();

    // 画像のマージンを取得
    console.log(`[PoseSet] finalize - Detecting image margins...`);
    let imageTrimming:
      | {
          marginTop: number;
          marginBottom: number;
          heightNew: number;
          heightOld: number;
          width: number;
        }
      | undefined = undefined;
    for (const pose of this.poses) {
      let imageTrimmer = new ImageTrimmer();
      if (!pose.frameImageDataUrl) {
        continue;
      }
      await imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);

      const marginColor = await imageTrimmer.getMarginColor();
      console.log(
        `[PoseSet] finalize - Detected margin color...`,
        pose.timeMiliseconds,
        marginColor
      );
      if (marginColor === null) continue;
      if (marginColor !== this.IMAGE_MARGIN_TRIMMING_COLOR) {
        continue;
      }
      const trimmed = await imageTrimmer.trimMargin(
        marginColor,
        this.IMAGE_MARGIN_TRIMMING_DIFF_THRESHOLD
      );
      if (!trimmed) continue;
      imageTrimming = trimmed;
      console.log(
        `[PoseSet] finalize - Determined image trimming positions...`,
        trimmed
      );
      break;
    }

    // 画像を整形
    for (const pose of this.poses) {
      let imageTrimmer = new ImageTrimmer();
      if (!pose.frameImageDataUrl || !pose.poseImageDataUrl) {
        continue;
      }

      console.log(
        `[PoseSet] finalize - Processing image...`,
        pose.timeMiliseconds
      );

      // 画像を整形 - フレーム画像
      await imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);

      if (imageTrimming) {
        await imageTrimmer.crop(
          0,
          imageTrimming.marginTop,
          imageTrimming.width,
          imageTrimming.heightNew
        );
      }

      await imageTrimmer.replaceColor(
        this.IMAGE_BACKGROUND_REPLACE_SRC_COLOR,
        this.IMAGE_BACKGROUND_REPLACE_DST_COLOR,
        this.IMAGE_BACKGROUND_REPLACE_DIFF_THRESHOLD
      );

      await imageTrimmer.resizeWithFit({
        width: this.IMAGE_WIDTH,
      });

      let newDataUrl = await imageTrimmer.getDataUrl(
        this.IMAGE_MIME,
        this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
          ? this.IMAGE_QUALITY
          : undefined
      );
      if (!newDataUrl) {
        console.warn(
          `[PoseSet] finalize - Could not get the new dataurl for frame image`
        );
        continue;
      }
      pose.frameImageDataUrl = newDataUrl;

      // 画像を整形 - ポーズプレビュー画像
      imageTrimmer = new ImageTrimmer();
      await imageTrimmer.loadByDataUrl(pose.poseImageDataUrl);

      if (imageTrimming) {
        await imageTrimmer.crop(
          0,
          imageTrimming.marginTop,
          imageTrimming.width,
          imageTrimming.heightNew
        );
      }

      await imageTrimmer.resizeWithFit({
        width: this.IMAGE_WIDTH,
      });

      newDataUrl = await imageTrimmer.getDataUrl(
        this.IMAGE_MIME,
        this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
          ? this.IMAGE_QUALITY
          : undefined
      );
      if (!newDataUrl) {
        console.warn(
          `[PoseSet] finalize - Could not get the new dataurl for pose preview image`
        );
        continue;
      }
      pose.poseImageDataUrl = newDataUrl;

      if (pose.faceFrameImageDataUrl) {
        // 画像を整形 - 顔フレーム画像
        imageTrimmer = new ImageTrimmer();
        await imageTrimmer.loadByDataUrl(pose.faceFrameImageDataUrl);

        newDataUrl = await imageTrimmer.getDataUrl(
          this.IMAGE_MIME,
          this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
            ? this.IMAGE_QUALITY
            : undefined
        );
        if (!newDataUrl) {
          console.warn(
            `[PoseSet] finalize - Could not get the new dataurl for face frame image`
          );
          continue;
        }
        pose.faceFrameImageDataUrl = newDataUrl;
      }
    }

    this.isFinalized = true;
  }

  removeDuplicatedPoses(): void {
    // 全ポーズを比較して類似ポーズを削除
    const newPoses: PoseSetItem[] = [];
    for (const poseA of this.poses) {
      let isDuplicated = false;
      for (const poseB of newPoses) {
        const isSimilarBodyPose = PoseSet.isSimilarBodyPose(
          poseA.bodyVectors,
          poseB.bodyVectors
        );
        const isSimilarHandPose =
          poseA.handVectors && poseB.handVectors
            ? PoseSet.isSimilarHandPose(poseA.handVectors, poseB.handVectors)
            : false;

        if (isSimilarBodyPose && isSimilarHandPose) {
          // 身体・手ともに類似ポーズならば
          isDuplicated = true;
          break;
        }
      }

      if (isDuplicated) continue;

      newPoses.push(poseA);
    }

    console.info(
      `[PoseSet] removeDuplicatedPoses - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`
    );
    this.poses = newPoses;
  }

  getSimilarPoses(
    results: Results,
    threshold: number = 0.9
  ): SimilarPoseItem[] {
    const bodyVector = PoseSet.getBodyVector((results as any).ea);
    if (!bodyVector) throw 'Could not get the body vector';

    const poses = [];
    for (const pose of this.poses) {
      const similarity = PoseSet.getBodyPoseSimilarity(
        pose.bodyVectors,
        bodyVector
      );
      if (threshold <= similarity) {
        poses.push({
          ...pose,
          similarity: similarity,
        });
      }
    }

    return poses;
  }

  static getBodyVector(
    poseLandmarks: { x: number; y: number; z: number }[]
  ): BodyVector | undefined {
    return {
      rightWristToRightElbow: [
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].x -
          poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].x,
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].y -
          poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].y,
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].z -
          poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].z,
      ],
      rightElbowToRightShoulder: [
        poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].x -
          poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER].x,
        poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].y -
          poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER].y,
        poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].z -
          poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER].z,
      ],
      leftWristToLeftElbow: [
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].x -
          poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].x,
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].y -
          poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].y,
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].z -
          poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].z,
      ],
      leftElbowToLeftShoulder: [
        poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].x -
          poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER].x,
        poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].y -
          poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER].y,
        poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].z -
          poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER].z,
      ],
    };
  }

  static getHandVectors(
    leftHandLandmarks: { x: number; y: number; z: number }[],
    rightHandLandmarks: { x: number; y: number; z: number }[]
  ): HandVector | undefined {
    if (
      (rightHandLandmarks === undefined || rightHandLandmarks.length === 0) &&
      (leftHandLandmarks === undefined || leftHandLandmarks.length === 0)
    ) {
      return undefined;
    }

    return {
      // 右手 - 親指
      rightThumbTipToFirstJoint:
        rightHandLandmarks === undefined || rightHandLandmarks.length === 0
          ? null
          : [
              rightHandLandmarks[4].x - rightHandLandmarks[3].x,
              rightHandLandmarks[4].y - rightHandLandmarks[3].y,
              rightHandLandmarks[4].z - rightHandLandmarks[3].z,
            ],
      rightThumbFirstJointToSecondJoint:
        rightHandLandmarks === undefined || rightHandLandmarks.length === 0
          ? null
          : [
              rightHandLandmarks[3].x - rightHandLandmarks[2].x,
              rightHandLandmarks[3].y - rightHandLandmarks[2].y,
              rightHandLandmarks[3].z - rightHandLandmarks[2].z,
            ],
      // 右手 - 人差し指
      rightIndexFingerTipToFirstJoint:
        rightHandLandmarks === undefined || rightHandLandmarks.length === 0
          ? null
          : [
              rightHandLandmarks[8].x - rightHandLandmarks[7].x,
              rightHandLandmarks[8].y - rightHandLandmarks[7].y,
              rightHandLandmarks[8].z - rightHandLandmarks[7].z,
            ],
      rightIndexFingerFirstJointToSecondJoint:
        rightHandLandmarks === undefined || rightHandLandmarks.length === 0
          ? null
          : [
              rightHandLandmarks[7].x - rightHandLandmarks[6].x,
              rightHandLandmarks[7].y - rightHandLandmarks[6].y,
              rightHandLandmarks[7].z - rightHandLandmarks[6].z,
            ],
      // 右手 - 中指
      rightMiddleFingerTipToFirstJoint:
        rightHandLandmarks === undefined || rightHandLandmarks.length === 0
          ? null
          : [
              rightHandLandmarks[12].x - rightHandLandmarks[11].x,
              rightHandLandmarks[12].y - rightHandLandmarks[11].y,
              rightHandLandmarks[12].z - rightHandLandmarks[11].z,
            ],
      rightMiddleFingerFirstJointToSecondJoint:
        rightHandLandmarks === undefined || rightHandLandmarks.length === 0
          ? null
          : [
              rightHandLandmarks[11].x - rightHandLandmarks[10].x,
              rightHandLandmarks[11].y - rightHandLandmarks[10].y,
              rightHandLandmarks[11].z - rightHandLandmarks[10].z,
            ],
      // 右手 - 薬指
      rightRingFingerTipToFirstJoint:
        rightHandLandmarks === undefined || rightHandLandmarks.length === 0
          ? null
          : [
              rightHandLandmarks[16].x - rightHandLandmarks[15].x,
              rightHandLandmarks[16].y - rightHandLandmarks[15].y,
              rightHandLandmarks[16].z - rightHandLandmarks[15].z,
            ],
      rightRingFingerFirstJointToSecondJoint:
        rightHandLandmarks === undefined || rightHandLandmarks.length === 0
          ? null
          : [
              rightHandLandmarks[15].x - rightHandLandmarks[14].x,
              rightHandLandmarks[15].y - rightHandLandmarks[14].y,
              rightHandLandmarks[15].z - rightHandLandmarks[14].z,
            ],
      // 右手 - 小指
      rightPinkyFingerTipToFirstJoint:
        rightHandLandmarks === undefined || rightHandLandmarks.length === 0
          ? null
          : [
              rightHandLandmarks[20].x - rightHandLandmarks[19].x,
              rightHandLandmarks[20].y - rightHandLandmarks[19].y,
              rightHandLandmarks[20].z - rightHandLandmarks[19].z,
            ],
      rightPinkyFingerFirstJointToSecondJoint:
        rightHandLandmarks === undefined || rightHandLandmarks.length === 0
          ? null
          : [
              rightHandLandmarks[19].x - rightHandLandmarks[18].x,
              rightHandLandmarks[19].y - rightHandLandmarks[18].y,
              rightHandLandmarks[19].z - rightHandLandmarks[18].z,
            ],
      // 左手 - 親指
      leftThumbTipToFirstJoint:
        leftHandLandmarks === undefined || leftHandLandmarks.length === 0
          ? null
          : [
              leftHandLandmarks[4].x - leftHandLandmarks[3].x,
              leftHandLandmarks[4].y - leftHandLandmarks[3].y,
              leftHandLandmarks[4].z - leftHandLandmarks[3].z,
            ],
      leftThumbFirstJointToSecondJoint:
        leftHandLandmarks === undefined || leftHandLandmarks.length === 0
          ? null
          : [
              leftHandLandmarks[3].x - leftHandLandmarks[2].x,
              leftHandLandmarks[3].y - leftHandLandmarks[2].y,
              leftHandLandmarks[3].z - leftHandLandmarks[2].z,
            ],
      // 左手 - 人差し指
      leftIndexFingerTipToFirstJoint:
        leftHandLandmarks === undefined || leftHandLandmarks.length === 0
          ? null
          : [
              leftHandLandmarks[8].x - leftHandLandmarks[7].x,
              leftHandLandmarks[8].y - leftHandLandmarks[7].y,
              leftHandLandmarks[8].z - leftHandLandmarks[7].z,
            ],
      leftIndexFingerFirstJointToSecondJoint:
        leftHandLandmarks === undefined || leftHandLandmarks.length === 0
          ? null
          : [
              leftHandLandmarks[7].x - leftHandLandmarks[6].x,
              leftHandLandmarks[7].y - leftHandLandmarks[6].y,
              leftHandLandmarks[7].z - leftHandLandmarks[6].z,
            ],
      // 左手 - 中指
      leftMiddleFingerTipToFirstJoint:
        leftHandLandmarks === undefined || leftHandLandmarks.length === 0
          ? null
          : [
              leftHandLandmarks[12].x - leftHandLandmarks[11].x,
              leftHandLandmarks[12].y - leftHandLandmarks[11].y,
              leftHandLandmarks[12].z - leftHandLandmarks[11].z,
            ],
      leftMiddleFingerFirstJointToSecondJoint:
        leftHandLandmarks === undefined || leftHandLandmarks.length === 0
          ? null
          : [
              leftHandLandmarks[11].x - leftHandLandmarks[10].x,
              leftHandLandmarks[11].y - leftHandLandmarks[10].y,
              leftHandLandmarks[11].z - leftHandLandmarks[10].z,
            ],
      // 左手 - 薬指
      leftRingFingerTipToFirstJoint:
        leftHandLandmarks === undefined || leftHandLandmarks.length === 0
          ? null
          : [
              leftHandLandmarks[16].x - leftHandLandmarks[15].x,
              leftHandLandmarks[16].y - leftHandLandmarks[15].y,
              leftHandLandmarks[16].z - leftHandLandmarks[15].z,
            ],
      leftRingFingerFirstJointToSecondJoint:
        leftHandLandmarks === undefined || leftHandLandmarks.length === 0
          ? null
          : [
              leftHandLandmarks[15].x - leftHandLandmarks[14].x,
              leftHandLandmarks[15].y - leftHandLandmarks[14].y,
              leftHandLandmarks[15].z - leftHandLandmarks[14].z,
            ],
      // 左手 - 小指
      leftPinkyFingerTipToFirstJoint:
        leftHandLandmarks === undefined || leftHandLandmarks.length === 0
          ? null
          : [
              leftHandLandmarks[20].x - leftHandLandmarks[19].x,
              leftHandLandmarks[20].y - leftHandLandmarks[19].y,
              leftHandLandmarks[20].z - leftHandLandmarks[19].z,
            ],
      leftPinkyFingerFirstJointToSecondJoint:
        leftHandLandmarks === undefined || leftHandLandmarks.length === 0
          ? null
          : [
              leftHandLandmarks[19].x - leftHandLandmarks[18].x,
              leftHandLandmarks[19].y - leftHandLandmarks[18].y,
              leftHandLandmarks[19].z - leftHandLandmarks[18].z,
            ],
    };
  }

  static isSimilarBodyPose(
    bodyVectorA: BodyVector,
    bodyVectorB: BodyVector,
    threshold = 0.8
  ): boolean {
    let isSimilar = false;
    const similarity = PoseSet.getBodyPoseSimilarity(bodyVectorA, bodyVectorB);
    if (similarity >= threshold) isSimilar = true;

    // console.log(`[PoseSet] isSimilarPose`, isSimilar, similarity);

    return isSimilar;
  }

  static getBodyPoseSimilarity(
    bodyVectorA: BodyVector,
    bodyVectorB: BodyVector
  ): number {
    const cosSimilarities = {
      leftWristToLeftElbow: cosSimilarity(
        bodyVectorA.leftWristToLeftElbow,
        bodyVectorB.leftWristToLeftElbow
      ),
      leftElbowToLeftShoulder: cosSimilarity(
        bodyVectorA.leftElbowToLeftShoulder,
        bodyVectorB.leftElbowToLeftShoulder
      ),
      rightWristToRightElbow: cosSimilarity(
        bodyVectorA.rightWristToRightElbow,
        bodyVectorB.rightWristToRightElbow
      ),
      rightElbowToRightShoulder: cosSimilarity(
        bodyVectorA.rightElbowToRightShoulder,
        bodyVectorB.rightElbowToRightShoulder
      ),
    };

    const cosSimilaritiesSum = Object.values(cosSimilarities).reduce(
      (sum, value) => sum + value,
      0
    );
    return cosSimilaritiesSum / Object.keys(cosSimilarities).length;
  }

  static isSimilarHandPose(
    handVectorA: HandVector,
    handVectorB: HandVector,
    threshold = 0.75
  ): boolean {
    const similarity = PoseSet.getHandSimilarity(handVectorA, handVectorB);
    if (similarity === -1) {
      return true;
    }
    return similarity >= threshold;
  }

  static getHandSimilarity(
    handVectorA: HandVector,
    handVectorB: HandVector
  ): number {
    const cosSimilaritiesRightHand =
      handVectorA.rightThumbFirstJointToSecondJoint === null ||
      handVectorB.rightThumbFirstJointToSecondJoint === null
        ? undefined
        : {
            // 右手 - 親指
            rightThumbTipToFirstJoint: cosSimilarity(
              handVectorA.rightThumbTipToFirstJoint,
              handVectorB.rightThumbTipToFirstJoint
            ),
            rightThumbFirstJointToSecondJoint: cosSimilarity(
              handVectorA.rightThumbFirstJointToSecondJoint,
              handVectorB.rightThumbFirstJointToSecondJoint
            ),
            // 右手 - 人差し指
            rightIndexFingerTipToFirstJoint: cosSimilarity(
              handVectorA.rightIndexFingerTipToFirstJoint,
              handVectorB.rightIndexFingerTipToFirstJoint
            ),
            rightIndexFingerFirstJointToSecondJoint: cosSimilarity(
              handVectorA.rightIndexFingerFirstJointToSecondJoint,
              handVectorB.rightIndexFingerFirstJointToSecondJoint
            ),
            // 右手 - 中指
            rightMiddleFingerTipToFirstJoint: cosSimilarity(
              handVectorA.rightMiddleFingerTipToFirstJoint,
              handVectorB.rightMiddleFingerTipToFirstJoint
            ),
            rightMiddleFingerFirstJointToSecondJoint: cosSimilarity(
              handVectorA.rightMiddleFingerFirstJointToSecondJoint,
              handVectorB.rightMiddleFingerFirstJointToSecondJoint
            ),
            // 右手 - 薬指
            rightRingFingerTipToFirstJoint: cosSimilarity(
              handVectorA.rightRingFingerTipToFirstJoint,
              handVectorB.rightRingFingerFirstJointToSecondJoint
            ),
            rightRingFingerFirstJointToSecondJoint: cosSimilarity(
              handVectorA.rightRingFingerFirstJointToSecondJoint,
              handVectorB.rightRingFingerFirstJointToSecondJoint
            ),
            // 右手 - 小指
            rightPinkyFingerTipToFirstJoint: cosSimilarity(
              handVectorA.rightPinkyFingerTipToFirstJoint,
              handVectorB.rightPinkyFingerTipToFirstJoint
            ),
            rightPinkyFingerFirstJointToSecondJoint: cosSimilarity(
              handVectorA.rightPinkyFingerFirstJointToSecondJoint,
              handVectorB.rightPinkyFingerFirstJointToSecondJoint
            ),
          };
    const cosSimilaritiesLeftHand =
      handVectorA.leftThumbFirstJointToSecondJoint === null ||
      handVectorB.leftThumbFirstJointToSecondJoint === null
        ? undefined
        : {
            // 左手 - 親指
            leftThumbTipToFirstJoint: cosSimilarity(
              handVectorA.leftThumbTipToFirstJoint,
              handVectorB.leftThumbTipToFirstJoint
            ),
            leftThumbFirstJointToSecondJoint: cosSimilarity(
              handVectorA.leftThumbFirstJointToSecondJoint,
              handVectorB.leftThumbFirstJointToSecondJoint
            ),
            // 左手 - 人差し指
            leftIndexFingerTipToFirstJoint: cosSimilarity(
              handVectorA.leftIndexFingerTipToFirstJoint,
              handVectorB.leftIndexFingerTipToFirstJoint
            ),
            leftIndexFingerFirstJointToSecondJoint: cosSimilarity(
              handVectorA.leftIndexFingerFirstJointToSecondJoint,
              handVectorB.leftIndexFingerFirstJointToSecondJoint
            ),
            // 左手 - 中指
            leftMiddleFingerTipToFirstJoint: cosSimilarity(
              handVectorA.leftMiddleFingerTipToFirstJoint,
              handVectorB.leftMiddleFingerTipToFirstJoint
            ),
            leftMiddleFingerFirstJointToSecondJoint: cosSimilarity(
              handVectorA.leftMiddleFingerFirstJointToSecondJoint,
              handVectorB.leftMiddleFingerFirstJointToSecondJoint
            ),
            // 左手 - 薬指
            leftRingFingerTipToFirstJoint: cosSimilarity(
              handVectorA.leftRingFingerTipToFirstJoint,
              handVectorB.leftRingFingerTipToFirstJoint
            ),
            leftRingFingerFirstJointToSecondJoint: cosSimilarity(
              handVectorA.leftRingFingerFirstJointToSecondJoint,
              handVectorB.leftRingFingerFirstJointToSecondJoint
            ),
            // 左手 - 小指
            leftPinkyFingerTipToFirstJoint: cosSimilarity(
              handVectorA.leftPinkyFingerTipToFirstJoint,
              handVectorB.leftPinkyFingerTipToFirstJoint
            ),
            leftPinkyFingerFirstJointToSecondJoint: cosSimilarity(
              handVectorA.leftPinkyFingerFirstJointToSecondJoint,
              handVectorB.leftPinkyFingerFirstJointToSecondJoint
            ),
          };

    let cosSimilaritiesSumLeftHand = 0;
    if (cosSimilaritiesLeftHand) {
      cosSimilaritiesSumLeftHand = Object.values(
        cosSimilaritiesLeftHand
      ).reduce((sum, value) => sum + value, 0);
    }

    let cosSimilaritiesSumRightHand = 0;
    if (cosSimilaritiesRightHand) {
      cosSimilaritiesSumRightHand = Object.values(
        cosSimilaritiesRightHand
      ).reduce((sum, value) => sum + value, 0);
    }

    if (cosSimilaritiesRightHand && cosSimilaritiesLeftHand) {
      return (
        (cosSimilaritiesSumRightHand + cosSimilaritiesSumLeftHand) /
        (Object.keys(cosSimilaritiesRightHand!).length +
          Object.keys(cosSimilaritiesLeftHand!).length)
      );
    } else if (cosSimilaritiesSumRightHand) {
      return (
        cosSimilaritiesSumRightHand /
        Object.keys(cosSimilaritiesRightHand!).length
      );
    } else if (cosSimilaritiesLeftHand) {
      return (
        cosSimilaritiesSumLeftHand /
        Object.keys(cosSimilaritiesLeftHand!).length
      );
    } else {
      return -1;
    }
  }

  public async getZip(): Promise<Blob> {
    const jsZip = new JSZip();
    jsZip.file('poses.json', await this.getJson());

    const imageFileExt = this.getFileExtensionByMime(this.IMAGE_MIME);

    for (const pose of this.poses) {
      if (pose.frameImageDataUrl) {
        try {
          const index =
            pose.frameImageDataUrl.indexOf('base64,') + 'base64,'.length;
          const base64 = pose.frameImageDataUrl.substring(index);
          jsZip.file(`frame-${pose.timeMiliseconds}.${imageFileExt}`, base64, {
            base64: true,
          });
        } catch (error) {
          console.warn(
            `[PoseExporterService] push - Could not push frame image`,
            error
          );
          throw error;
        }
      }
      if (pose.poseImageDataUrl) {
        try {
          const index =
            pose.poseImageDataUrl.indexOf('base64,') + 'base64,'.length;
          const base64 = pose.poseImageDataUrl.substring(index);
          jsZip.file(`pose-${pose.timeMiliseconds}.${imageFileExt}`, base64, {
            base64: true,
          });
        } catch (error) {
          console.warn(
            `[PoseExporterService] push - Could not push frame image`,
            error
          );
          throw error;
        }
      }
      if (pose.faceFrameImageDataUrl) {
        try {
          const index =
            pose.faceFrameImageDataUrl.indexOf('base64,') + 'base64,'.length;
          const base64 = pose.faceFrameImageDataUrl.substring(index);
          jsZip.file(`face-${pose.timeMiliseconds}.${imageFileExt}`, base64, {
            base64: true,
          });
        } catch (error) {
          console.warn(
            `[PoseExporterService] push - Could not push face frame image`,
            error
          );
          throw error;
        }
      }
    }

    return await jsZip.generateAsync({ type: 'blob' });
  }

  getFileExtensionByMime(IMAGE_MIME: string) {
    switch (IMAGE_MIME) {
      case 'image/png':
        return 'png';
      case 'image/jpeg':
        return 'jpg';
      case 'image/webp':
        return 'webp';
      default:
        return 'png';
    }
  }

  public async getJson(): Promise<string> {
    if (this.videoMetadata === undefined || this.poses === undefined)
      return '{}';

    if (!this.isFinalized) {
      await this.finalize();
    }

    let poseLandmarkMappings = [];
    for (const key of Object.keys(POSE_LANDMARKS)) {
      const index: number = POSE_LANDMARKS[key as keyof typeof POSE_LANDMARKS];
      poseLandmarkMappings[index] = key;
    }

    const json: PoseSetJson = {
      generator: 'mp-video-pose-extractor',
      version: 1,
      video: this.videoMetadata!,
      poses: this.poses.map((pose: PoseSetItem): PoseSetJsonItem => {
        // BodyVector の圧縮
        const bodyVector = [];
        for (const key of PoseSet.BODY_VECTOR_MAPPINGS) {
          bodyVector.push(pose.bodyVectors[key as keyof BodyVector]);
        }

        // HandVector の圧縮
        let handVector: (number[] | null)[] | undefined = undefined;
        if (pose.handVectors) {
          handVector = [];
          for (const key of PoseSet.HAND_VECTOR_MAPPINGS) {
            handVector.push(pose.handVectors[key as keyof HandVector]);
          }
        }

        // PoseSetJsonItem の pose オブジェクトを生成
        return {
          t: pose.timeMiliseconds,
          d: pose.durationMiliseconds,
          p: pose.pose,
          l: pose.leftHand,
          r: pose.rightHand,
          v: bodyVector,
          h: handVector,
          e: pose.extendedData,
        };
      }),
      poseLandmarkMapppings: poseLandmarkMappings,
    };

    return JSON.stringify(json);
  }

  loadJson(json: string | any) {
    const parsedJson = typeof json === 'string' ? JSON.parse(json) : json;

    if (parsedJson.generator !== 'mp-video-pose-extractor') {
      throw '不正なファイル';
    } else if (parsedJson.version !== 1) {
      throw '未対応のバージョン';
    }

    this.videoMetadata = parsedJson.video;
    this.poses = parsedJson.poses.map((item: PoseSetJsonItem): PoseSetItem => {
      const bodyVector: any = {};
      PoseSet.BODY_VECTOR_MAPPINGS.map((key, index) => {
        bodyVector[key as keyof BodyVector] = item.v[index];
      });

      const handVector: any = {};
      if (item.h) {
        PoseSet.HAND_VECTOR_MAPPINGS.map((key, index) => {
          handVector[key as keyof HandVector] = item.h![index];
        });
      }

      return {
        timeMiliseconds: item.t,
        durationMiliseconds: item.d,
        pose: item.p,
        leftHand: item.l,
        rightHand: item.r,
        bodyVectors: bodyVector,
        handVectors: handVector,
        frameImageDataUrl: undefined,
        extendedData: item.e,
      };
    });
  }

  async loadZip(buffer: ArrayBuffer, includeImages: boolean = true) {
    console.log(`[PoseSet] loadZip...`, JSZip);
    const jsZip = new JSZip();
    console.log(`[PoseSet] init...`);
    const zip = await jsZip.loadAsync(buffer, { base64: false });
    if (!zip) throw 'ZIPファイルを読み込めませんでした';

    const json = await zip.file('poses.json')?.async('text');
    if (json === undefined) {
      throw 'ZIPファイルに pose.json が含まれていません';
    }

    this.loadJson(json);

    const fileExt = this.getFileExtensionByMime(this.IMAGE_MIME);

    if (includeImages) {
      for (const pose of this.poses) {
        if (!pose.frameImageDataUrl) {
          const frameImageFileName = `frame-${pose.timeMiliseconds}.${fileExt}`;
          const imageBase64 = await zip
            .file(frameImageFileName)
            ?.async('base64');
          if (imageBase64) {
            pose.frameImageDataUrl = `data:${this.IMAGE_MIME};base64,${imageBase64}`;
          }
        }
        if (!pose.poseImageDataUrl) {
          const poseImageFileName = `pose-${pose.timeMiliseconds}.${fileExt}`;
          const imageBase64 = await zip
            .file(poseImageFileName)
            ?.async('base64');
          if (imageBase64) {
            pose.poseImageDataUrl = `data:${this.IMAGE_MIME};base64,${imageBase64}`;
          }
        }
      }
    }
  }
}
