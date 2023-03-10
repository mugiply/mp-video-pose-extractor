import { POSE_LANDMARKS, Results } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
import { PoseSetItem } from '../interfaces/pose-set-item';
import { PoseSetJson } from '../interfaces/pose-set-json';
import { PoseSetJsonItem } from '../interfaces/pose-set-json-item';
import { BodyVector } from '../interfaces/body-vector';

// @ts-ignore
import cosSimilarityA from 'cos-similarity';
// @ts-ignore
import * as cosSimilarityB from 'cos-similarity';

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
  ];

  // ポーズを追加するためのキュー
  private similarPoseQueue: PoseSetItem[] = [];

  // 類似ポーズの除去 - 各ポーズの前後から
  private readonly IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_AROUND = true;

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

  /**
   * ポーズ数の取得
   * @returns
   */
  getNumberOfPoses(): number {
    if (this.poses === undefined) return -1;
    return this.poses.length;
  }

  /**
   * 全ポーズの取得
   * @returns 全てのポーズ
   */
  getPoses(): PoseSetItem[] {
    if (this.poses === undefined) return [];
    return this.poses;
  }

  /**
   * 指定されたID (PoseSetItemId) によるポーズの取得
   * @param poseSetItemId
   * @returns ポーズ
   */
  getPoseById(poseSetItemId: number): PoseSetItem | undefined {
    if (this.poses === undefined) return undefined;
    return this.poses.find((pose) => pose.id === poseSetItemId);
  }

  /**
   * 指定された時間によるポーズの取得
   * @param timeMiliseconds ポーズの時間 (ミリ秒)
   * @returns ポーズ
   */
  getPoseByTime(timeMiliseconds: number): PoseSetItem | undefined {
    if (this.poses === undefined) return undefined;
    return this.poses.find((pose) => pose.timeMiliseconds === timeMiliseconds);
  }

  /**
   * ポーズの追加
   */
  pushPose(
    videoTimeMiliseconds: number,
    frameImageDataUrl: string | undefined,
    poseImageDataUrl: string | undefined,
    faceFrameImageDataUrl: string | undefined,
    results: Results
  ): PoseSetItem | undefined {
    if (this.poses.length === 0) {
      this.videoMetadata.firstPoseDetectedTime = videoTimeMiliseconds;
    }

    if (results.poseLandmarks === undefined) {
      console.debug(
        `[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get poseLandmarks`,
        results
      );
      return;
    }

    const poseLandmarksWithWorldCoordinate: any[] = (results as any).ea
      ? (results as any).ea
      : [];
    if (poseLandmarksWithWorldCoordinate.length === 0) {
      console.debug(
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
      console.debug(
        `[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the hand landmarks`,
        results
      );
    } else if (results.leftHandLandmarks === undefined) {
      console.debug(
        `[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the left hand landmarks`,
        results
      );
    } else if (results.rightHandLandmarks === undefined) {
      console.debug(
        `[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the right hand landmarks`,
        results
      );
    }

    const handVector = PoseSet.getHandVector(
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
      id: PoseSet.getIdByTimeMiliseconds(videoTimeMiliseconds),
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
      rightHand: results.rightHandLandmarks?.map((normalizedLandmark) => {
        return [
          normalizedLandmark.x,
          normalizedLandmark.y,
          normalizedLandmark.z,
        ];
      }),
      bodyVector: bodyVector,
      handVector: handVector,
      frameImageDataUrl: frameImageDataUrl,
      poseImageDataUrl: poseImageDataUrl,
      faceFrameImageDataUrl: faceFrameImageDataUrl,
      extendedData: {},
      debug: {
        duplicatedItems: [],
      },
      mergedTimeMiliseconds: videoTimeMiliseconds,
      mergedDurationMiliseconds: -1,
    };

    let lastPose;
    if (this.poses.length === 0 && 1 <= this.similarPoseQueue.length) {
      // 類似ポーズキューから最後のポーズを取得
      lastPose = this.similarPoseQueue[this.similarPoseQueue.length - 1];
    } else if (1 <= this.poses.length) {
      // ポーズ配列から最後のポーズを取得
      lastPose = this.poses[this.poses.length - 1];
    }

    if (lastPose) {
      // 最後のポーズがあれば、類似ポーズかどうかを比較
      const isSimilarBodyPose = PoseSet.isSimilarBodyPose(
        pose.bodyVector,
        lastPose.bodyVector
      );

      let isSimilarHandPose = true;
      if (lastPose.handVector && pose.handVector) {
        isSimilarHandPose = PoseSet.isSimilarHandPose(
          pose.handVector,
          lastPose.handVector
        );
      } else if (!lastPose.handVector && pose.handVector) {
        isSimilarHandPose = false;
      }

      if (!isSimilarBodyPose || !isSimilarHandPose) {
        // 身体・手のいずれかが前のポーズと類似していないならば、類似ポーズキューを処理して、ポーズ配列へ追加
        this.pushPoseFromSimilarPoseQueue(pose.timeMiliseconds);
      }
    }

    // 類似ポーズキューへ追加
    this.similarPoseQueue.push(pose);

    return pose;
  }

  /**
   * ポーズの配列からポーズが決まっている瞬間を取得
   * @param poses ポーズの配列
   * @returns ポーズが決まっている瞬間
   */
  static getSuitablePoseByPoses(poses: PoseSetItem[]): PoseSetItem {
    if (poses.length === 0) return null;
    if (poses.length === 1) {
      return poses[1];
    }

    // 各標本ポーズごとの類似度を初期化
    const similaritiesOfPoses: {
      [key: number]: {
        handSimilarity: number;
        bodySimilarity: number;
      }[];
    } = {};
    for (let i = 0; i < poses.length; i++) {
      similaritiesOfPoses[poses[i].timeMiliseconds] = poses.map(
        (pose: PoseSetItem) => {
          return {
            handSimilarity: 0,
            bodySimilarity: 0,
          };
        }
      );
    }

    // 各標本ポーズごとの類似度を計算
    for (let samplePose of poses) {
      let handSimilarity: number;

      for (let i = 0; i < poses.length; i++) {
        const pose = poses[i];
        if (pose.handVector && samplePose.handVector) {
          handSimilarity = PoseSet.getHandSimilarity(
            pose.handVector,
            samplePose.handVector
          );
        }

        let bodySimilarity = PoseSet.getBodyPoseSimilarity(
          pose.bodyVector,
          samplePose.bodyVector
        );

        similaritiesOfPoses[samplePose.timeMiliseconds][i] = {
          handSimilarity: handSimilarity ?? 0,
          bodySimilarity,
        };
      }
    }

    // 類似度の高いフレームが多かったポーズを選択
    const similaritiesOfSamplePoses = poses.map((pose: PoseSetItem) => {
      return similaritiesOfPoses[pose.timeMiliseconds].reduce(
        (
          prev: number,
          current: { handSimilarity: number; bodySimilarity: number }
        ) => {
          return prev + current.handSimilarity + current.bodySimilarity;
        },
        0
      );
    });
    const maxSimilarity = Math.max(...similaritiesOfSamplePoses);
    const maxSimilarityIndex = similaritiesOfSamplePoses.indexOf(maxSimilarity);
    const selectedPose = poses[maxSimilarityIndex];
    if (!selectedPose) {
      console.warn(
        `[PoseSet] getSuitablePoseByPoses`,
        similaritiesOfSamplePoses,
        maxSimilarity,
        maxSimilarityIndex
      );
    }

    console.debug(`[PoseSet] getSuitablePoseByPoses`, {
      selected: selectedPose,
      unselected: poses.filter((pose: PoseSetItem) => {
        return pose.timeMiliseconds !== selectedPose.timeMiliseconds;
      }),
    });
    return selectedPose;
  }

  /**
   * 最終処理
   * (重複したポーズの除去、画像のマージン除去など)
   */
  async finalize(isRemoveDuplicate: boolean = true) {
    if (this.similarPoseQueue.length > 0) {
      // 類似ポーズキューにポーズが残っている場合、最適なポーズを選択してポーズ配列へ追加
      this.pushPoseFromSimilarPoseQueue(this.videoMetadata.duration);
    }

    if (0 == this.poses.length) {
      // ポーズが一つもない場合、処理を終了
      this.isFinalized = true;
      return;
    }

    // ポーズの持続時間を設定
    for (let i = 0; i < this.poses.length - 1; i++) {
      if (this.poses[i].durationMiliseconds === -1) {
        this.poses[i].durationMiliseconds =
          this.poses[i + 1].timeMiliseconds - this.poses[i].timeMiliseconds;
      }
      if (this.poses[i].mergedDurationMiliseconds === -1) {
        this.poses[i].mergedDurationMiliseconds =
          this.poses[i].durationMiliseconds;
      }
    }

    if (this.poses[this.poses.length - 1].durationMiliseconds === -1) {
      this.poses[this.poses.length - 1].durationMiliseconds =
        this.videoMetadata.duration -
        this.poses[this.poses.length - 1].timeMiliseconds;
    }
    if (this.poses[this.poses.length - 1].mergedDurationMiliseconds === -1) {
      this.poses[this.poses.length - 1].mergedDurationMiliseconds =
        this.poses[this.poses.length - 1].durationMiliseconds;
    }

    // 全体から重複ポーズを除去
    if (isRemoveDuplicate) {
      this.removeDuplicatedPoses();
    }

    // 最初のポーズを除去
    this.poses.shift();

    // 画像のマージンを取得
    console.debug(`[PoseSet] finalize - Detecting image margins...`);
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
      console.debug(
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
      console.debug(
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

      console.debug(
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

  /**
   * 類似ポーズの取得
   * @param results MediaPipe Holistic によるポーズの検出結果
   * @param threshold しきい値
   * @param targetRange ポーズを比較する範囲 (all: 全て, bodyPose: 身体のみ, handPose: 手指のみ)
   * @returns 類似ポーズの配列
   */
  getSimilarPoses(
    results: Results,
    threshold: number = 0.9,
    targetRange: 'all' | 'bodyPose' | 'handPose' = 'all'
  ): SimilarPoseItem[] {
    // 身体のベクトルを取得
    let bodyVector: BodyVector;
    try {
      bodyVector = PoseSet.getBodyVector((results as any).ea);
    } catch (e) {
      console.error(`[PoseSet] getSimilarPoses - Error occurred`, e, results);
      return [];
    }
    if (!bodyVector) {
      throw 'Could not get the body vector';
    }

    // 手指のベクトルを取得
    let handVector: HandVector;
    if (targetRange === 'all' || targetRange === 'handPose') {
      handVector = PoseSet.getHandVector(
        results.leftHandLandmarks,
        results.rightHandLandmarks
      );
      if (targetRange === 'handPose' && !handVector) {
        throw 'Could not get the hand vector';
      }
    }

    // 各ポーズとベクトルを比較
    const poses = [];
    for (const pose of this.poses) {
      if (
        (targetRange === 'all' || targetRange === 'bodyPose') &&
        !pose.bodyVector
      ) {
        continue;
      } else if (targetRange === 'handPose' && !pose.handVector) {
        continue;
      }

      /*console.debug(
        '[PoseSet] getSimilarPoses - ',
        this.getVideoName(),
        pose.timeMiliseconds
      );*/

      // 身体のポーズの類似度を取得
      let bodySimilarity: number;
      if (bodyVector && pose.bodyVector) {
        bodySimilarity = PoseSet.getBodyPoseSimilarity(
          pose.bodyVector,
          bodyVector
        );
      }

      // 手指のポーズの類似度を取得
      let handSimilarity: number;
      if (handVector && pose.handVector) {
        handSimilarity = PoseSet.getHandSimilarity(pose.handVector, handVector);
      }

      // 判定
      let similarity: number,
        isSimilar = false;
      if (targetRange === 'all') {
        similarity = Math.max(bodySimilarity ?? 0, handSimilarity ?? 0);
        if (threshold <= bodySimilarity || threshold <= handSimilarity) {
          isSimilar = true;
        }
      } else if (targetRange === 'bodyPose') {
        similarity = bodySimilarity;
        if (threshold <= bodySimilarity) {
          isSimilar = true;
        }
      } else if (targetRange === 'handPose') {
        similarity = handSimilarity;
        if (threshold <= handSimilarity) {
          isSimilar = true;
        }
      }

      if (!isSimilar) continue;

      // 結果へ追加
      poses.push({
        ...pose,
        similarity: similarity,
        bodyPoseSimilarity: bodySimilarity,
        handPoseSimilarity: handSimilarity,
      } as SimilarPoseItem);
    }

    return poses;
  }

  /**
   * 身体の姿勢を表すベクトルの取得
   * @param poseLandmarks MediaPipe Holistic で取得できた身体のワールド座標 (ra 配列)
   * @returns ベクトル
   */
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

  /**
   * 手指の姿勢を表すベクトルの取得
   * @param leftHandLandmarks MediaPipe Holistic で取得できた左手の正規化座標
   * @param rightHandLandmarks MediaPipe Holistic で取得できた右手の正規化座標
   * @returns ベクトル
   */
  static getHandVector(
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

  /**
   * BodyVector 間が類似しているかどうかの判定
   * @param bodyVectorA 比較先の BodyVector
   * @param bodyVectorB 比較元の BodyVector
   * @param threshold しきい値
   * @returns 類似しているかどうか
   */
  static isSimilarBodyPose(
    bodyVectorA: BodyVector,
    bodyVectorB: BodyVector,
    threshold = 0.8
  ): boolean {
    let isSimilar = false;
    const similarity = PoseSet.getBodyPoseSimilarity(bodyVectorA, bodyVectorB);
    if (similarity >= threshold) isSimilar = true;

    // console.debug(`[PoseSet] isSimilarPose`, isSimilar, similarity);

    return isSimilar;
  }

  /**
   * 身体ポーズの類似度の取得
   * @param bodyVectorA 比較先の BodyVector
   * @param bodyVectorB 比較元の BodyVector
   * @returns 類似度
   */
  static getBodyPoseSimilarity(
    bodyVectorA: BodyVector,
    bodyVectorB: BodyVector
  ): number {
    const cosSimilarities = {
      leftWristToLeftElbow: PoseSet.getCosSimilarity(
        bodyVectorA.leftWristToLeftElbow,
        bodyVectorB.leftWristToLeftElbow
      ),
      leftElbowToLeftShoulder: PoseSet.getCosSimilarity(
        bodyVectorA.leftElbowToLeftShoulder,
        bodyVectorB.leftElbowToLeftShoulder
      ),
      rightWristToRightElbow: PoseSet.getCosSimilarity(
        bodyVectorA.rightWristToRightElbow,
        bodyVectorB.rightWristToRightElbow
      ),
      rightElbowToRightShoulder: PoseSet.getCosSimilarity(
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

  /**
   * HandVector 間が類似しているかどうかの判定
   * @param handVectorA 比較先の HandVector
   * @param handVectorB 比較元の HandVector
   * @param threshold しきい値
   * @returns 類似しているかどうか
   */
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

  /**
   * 手のポーズの類似度の取得
   * @param handVectorA 比較先の HandVector
   * @param handVectorB 比較元の HandVector
   * @returns 類似度
   */
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
            rightThumbTipToFirstJoint: PoseSet.getCosSimilarity(
              handVectorA.rightThumbTipToFirstJoint,
              handVectorB.rightThumbTipToFirstJoint
            ),
            rightThumbFirstJointToSecondJoint: PoseSet.getCosSimilarity(
              handVectorA.rightThumbFirstJointToSecondJoint,
              handVectorB.rightThumbFirstJointToSecondJoint
            ),
            // 右手 - 人差し指
            rightIndexFingerTipToFirstJoint: PoseSet.getCosSimilarity(
              handVectorA.rightIndexFingerTipToFirstJoint,
              handVectorB.rightIndexFingerTipToFirstJoint
            ),
            rightIndexFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(
              handVectorA.rightIndexFingerFirstJointToSecondJoint,
              handVectorB.rightIndexFingerFirstJointToSecondJoint
            ),
            // 右手 - 中指
            rightMiddleFingerTipToFirstJoint: PoseSet.getCosSimilarity(
              handVectorA.rightMiddleFingerTipToFirstJoint,
              handVectorB.rightMiddleFingerTipToFirstJoint
            ),
            rightMiddleFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(
              handVectorA.rightMiddleFingerFirstJointToSecondJoint,
              handVectorB.rightMiddleFingerFirstJointToSecondJoint
            ),
            // 右手 - 薬指
            rightRingFingerTipToFirstJoint: PoseSet.getCosSimilarity(
              handVectorA.rightRingFingerTipToFirstJoint,
              handVectorB.rightRingFingerFirstJointToSecondJoint
            ),
            rightRingFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(
              handVectorA.rightRingFingerFirstJointToSecondJoint,
              handVectorB.rightRingFingerFirstJointToSecondJoint
            ),
            // 右手 - 小指
            rightPinkyFingerTipToFirstJoint: PoseSet.getCosSimilarity(
              handVectorA.rightPinkyFingerTipToFirstJoint,
              handVectorB.rightPinkyFingerTipToFirstJoint
            ),
            rightPinkyFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(
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
            leftThumbTipToFirstJoint: PoseSet.getCosSimilarity(
              handVectorA.leftThumbTipToFirstJoint,
              handVectorB.leftThumbTipToFirstJoint
            ),
            leftThumbFirstJointToSecondJoint: PoseSet.getCosSimilarity(
              handVectorA.leftThumbFirstJointToSecondJoint,
              handVectorB.leftThumbFirstJointToSecondJoint
            ),
            // 左手 - 人差し指
            leftIndexFingerTipToFirstJoint: PoseSet.getCosSimilarity(
              handVectorA.leftIndexFingerTipToFirstJoint,
              handVectorB.leftIndexFingerTipToFirstJoint
            ),
            leftIndexFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(
              handVectorA.leftIndexFingerFirstJointToSecondJoint,
              handVectorB.leftIndexFingerFirstJointToSecondJoint
            ),
            // 左手 - 中指
            leftMiddleFingerTipToFirstJoint: PoseSet.getCosSimilarity(
              handVectorA.leftMiddleFingerTipToFirstJoint,
              handVectorB.leftMiddleFingerTipToFirstJoint
            ),
            leftMiddleFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(
              handVectorA.leftMiddleFingerFirstJointToSecondJoint,
              handVectorB.leftMiddleFingerFirstJointToSecondJoint
            ),
            // 左手 - 薬指
            leftRingFingerTipToFirstJoint: PoseSet.getCosSimilarity(
              handVectorA.leftRingFingerTipToFirstJoint,
              handVectorB.leftRingFingerTipToFirstJoint
            ),
            leftRingFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(
              handVectorA.leftRingFingerFirstJointToSecondJoint,
              handVectorB.leftRingFingerFirstJointToSecondJoint
            ),
            // 左手 - 小指
            leftPinkyFingerTipToFirstJoint: PoseSet.getCosSimilarity(
              handVectorA.leftPinkyFingerTipToFirstJoint,
              handVectorB.leftPinkyFingerTipToFirstJoint
            ),
            leftPinkyFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(
              handVectorA.leftPinkyFingerFirstJointToSecondJoint,
              handVectorB.leftPinkyFingerFirstJointToSecondJoint
            ),
          };

    // 左手の類似度
    let cosSimilaritiesSumLeftHand = 0;
    if (cosSimilaritiesLeftHand) {
      cosSimilaritiesSumLeftHand = Object.values(
        cosSimilaritiesLeftHand
      ).reduce((sum, value) => sum + value, 0);
    }

    // 右手の類似度
    let cosSimilaritiesSumRightHand = 0;
    if (cosSimilaritiesRightHand) {
      cosSimilaritiesSumRightHand = Object.values(
        cosSimilaritiesRightHand
      ).reduce((sum, value) => sum + value, 0);
    }

    // 合算された類似度
    if (cosSimilaritiesRightHand && cosSimilaritiesLeftHand) {
      return (
        (cosSimilaritiesSumRightHand + cosSimilaritiesSumLeftHand) /
        (Object.keys(cosSimilaritiesRightHand!).length +
          Object.keys(cosSimilaritiesLeftHand!).length)
      );
    } else if (cosSimilaritiesRightHand) {
      if (
        handVectorB.leftThumbFirstJointToSecondJoint !== null &&
        handVectorA.leftThumbFirstJointToSecondJoint === null
      ) {
        // handVectorB で左手があるのに handVectorA で左手がない場合、類似度を減らす
        console.debug(
          `[PoseSet] getHandSimilarity - Adjust similarity, because left hand not found...`
        );
        return (
          cosSimilaritiesSumRightHand /
          (Object.keys(cosSimilaritiesRightHand!).length * 2)
        );
      }
      return (
        cosSimilaritiesSumRightHand /
        Object.keys(cosSimilaritiesRightHand!).length
      );
    } else if (cosSimilaritiesLeftHand) {
      if (
        handVectorB.rightThumbFirstJointToSecondJoint !== null &&
        handVectorA.rightThumbFirstJointToSecondJoint === null
      ) {
        // handVectorB で右手があるのに handVectorA で右手がない場合、類似度を減らす
        console.debug(
          `[PoseSet] getHandSimilarity - Adjust similarity, because right hand not found...`
        );
        return (
          cosSimilaritiesSumLeftHand /
          (Object.keys(cosSimilaritiesLeftHand!).length * 2)
        );
      }
      return (
        cosSimilaritiesSumLeftHand /
        Object.keys(cosSimilaritiesLeftHand!).length
      );
    }

    return -1;
  }

  /**
   * ZIP ファイルとしてのシリアライズ
   * @returns ZIPファイル (Blob 形式)
   */
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
          jsZip.file(`frame-${pose.id}.${imageFileExt}`, base64, {
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
          jsZip.file(`pose-${pose.id}.${imageFileExt}`, base64, {
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
          jsZip.file(`face-${pose.id}.${imageFileExt}`, base64, {
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

  /**
   * JSON 文字列としてのシリアライズ
   * @returns JSON 文字列
   */
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
          bodyVector.push(pose.bodyVector[key as keyof BodyVector]);
        }

        // HandVector の圧縮
        let handVector: (number[] | null)[] | undefined = undefined;
        if (pose.handVector) {
          handVector = [];
          for (const key of PoseSet.HAND_VECTOR_MAPPINGS) {
            handVector.push(pose.handVector[key as keyof HandVector]);
          }
        }

        // PoseSetJsonItem の pose オブジェクトを生成
        return {
          id: pose.id,
          t: pose.timeMiliseconds,
          d: pose.durationMiliseconds,
          p: pose.pose,
          l: pose.leftHand,
          r: pose.rightHand,
          v: bodyVector,
          h: handVector,
          e: pose.extendedData,
          md: pose.mergedDurationMiliseconds,
          mt: pose.mergedTimeMiliseconds,
        };
      }),
      poseLandmarkMapppings: poseLandmarkMappings,
    };

    return JSON.stringify(json);
  }

  /**
   * JSON からの読み込み
   * @param json JSON 文字列 または JSON オブジェクト
   */
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
        id:
          item.id === undefined
            ? PoseSet.getIdByTimeMiliseconds(item.t)
            : item.id,
        timeMiliseconds: item.t,
        durationMiliseconds: item.d,
        pose: item.p,
        leftHand: item.l,
        rightHand: item.r,
        bodyVector: bodyVector,
        handVector: handVector,
        frameImageDataUrl: undefined,
        extendedData: item.e,
        debug: undefined,
        mergedDurationMiliseconds: item.md,
        mergedTimeMiliseconds: item.mt,
      };
    });
  }

  /**
   * ZIP ファイルからの読み込み
   * @param buffer ZIP ファイルの Buffer
   * @param includeImages 画像を展開するかどうか
   */
  async loadZip(buffer: ArrayBuffer, includeImages: boolean = true) {
    const jsZip = new JSZip();
    console.debug(`[PoseSet] init...`);
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
          let imageBase64: string;
          if (zip.file(`frame-${pose.id}.${fileExt}`)) {
            imageBase64 = await zip
              .file(`frame-${pose.id}.${fileExt}`)
              ?.async('base64');
          } else {
            imageBase64 = await zip
              .file(`frame-${pose.timeMiliseconds}.${fileExt}`)
              ?.async('base64');
          }
          if (imageBase64) {
            pose.frameImageDataUrl = `data:${this.IMAGE_MIME};base64,${imageBase64}`;
          }
        }
        if (!pose.poseImageDataUrl) {
          const poseImageFileName = `pose-${pose.id}.${fileExt}`;
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

  static getCosSimilarity(a: number[], b: number[]) {
    if (cosSimilarityA) {
      return cosSimilarityA(a, b);
    }
    return cosSimilarityB(a, b);
  }

  private pushPoseFromSimilarPoseQueue(nextPoseTimeMiliseconds?: number) {
    if (this.similarPoseQueue.length === 0) return;

    if (this.similarPoseQueue.length === 1) {
      // 類似ポーズキューにポーズが一つしかない場合、当該ポーズをポーズ配列へ追加
      const pose = this.similarPoseQueue[0];
      this.poses.push(pose);
      this.similarPoseQueue = [];
      return;
    }

    // 各ポーズの持続時間を設定
    for (let i = 0; i < this.similarPoseQueue.length - 1; i++) {
      this.similarPoseQueue[i].durationMiliseconds =
        this.similarPoseQueue[i + 1].timeMiliseconds -
        this.similarPoseQueue[i].timeMiliseconds;
    }
    if (nextPoseTimeMiliseconds) {
      this.similarPoseQueue[
        this.similarPoseQueue.length - 1
      ].durationMiliseconds =
        nextPoseTimeMiliseconds -
        this.similarPoseQueue[this.similarPoseQueue.length - 1].timeMiliseconds;
    }

    // 類似ポーズキューの中から最も持続時間が長いポーズを選択
    const selectedPose = PoseSet.getSuitablePoseByPoses(this.similarPoseQueue);

    // 選択されなかったポーズを列挙
    selectedPose.debug.duplicatedItems = this.similarPoseQueue
      .filter((item: PoseSetItem) => {
        return item.timeMiliseconds !== selectedPose.timeMiliseconds;
      })
      .map((item: PoseSetItem) => {
        return {
          id: item.id,
          timeMiliseconds: item.timeMiliseconds,
          durationMiliseconds: item.durationMiliseconds,
        };
      });

    // 選択されたポーズの情報を更新
    selectedPose.mergedTimeMiliseconds =
      this.similarPoseQueue[0].timeMiliseconds;
    selectedPose.mergedDurationMiliseconds = this.similarPoseQueue.reduce(
      (sum: number, item: PoseSetItem) => {
        return sum + item.durationMiliseconds;
      },
      0
    );
    selectedPose.id = PoseSet.getIdByTimeMiliseconds(
      selectedPose.mergedTimeMiliseconds
    );

    // 当該ポーズをポーズ配列へ追加
    if (this.IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_AROUND) {
      this.poses.push(selectedPose);
    } else {
      // デバッグ用
      this.poses.push(...this.similarPoseQueue);
    }

    // 類似ポーズキューをクリア
    this.similarPoseQueue = [];
  }

  removeDuplicatedPoses(): void {
    // 全ポーズを比較して類似ポーズを削除
    const newPoses: PoseSetItem[] = [],
      removedPoses: PoseSetItem[] = [];
    for (const pose of this.poses) {
      let duplicatedPose: PoseSetItem;
      for (const insertedPose of newPoses) {
        const isSimilarBodyPose = PoseSet.isSimilarBodyPose(
          pose.bodyVector,
          insertedPose.bodyVector
        );
        const isSimilarHandPose =
          pose.handVector && insertedPose.handVector
            ? PoseSet.isSimilarHandPose(
                pose.handVector,
                insertedPose.handVector,
                0.9
              )
            : false;

        if (isSimilarBodyPose && isSimilarHandPose) {
          // 身体・手ともに類似ポーズならば
          duplicatedPose = insertedPose;
          break;
        }
      }

      if (duplicatedPose) {
        removedPoses.push(pose);
        if (duplicatedPose.debug.duplicatedItems) {
          duplicatedPose.debug.duplicatedItems.push({
            id: pose.id,
            timeMiliseconds: pose.timeMiliseconds,
            durationMiliseconds: pose.durationMiliseconds,
          });
        }
        continue;
      }

      newPoses.push(pose);
    }

    console.info(
      `[PoseSet] removeDuplicatedPoses - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`,
      {
        removed: removedPoses,
        keeped: newPoses,
      }
    );
    this.poses = newPoses;
  }

  static getIdByTimeMiliseconds(timeMiliseconds: number) {
    return Math.floor(timeMiliseconds / 100) * 100;
  }

  private getFileExtensionByMime(IMAGE_MIME: string) {
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
}
