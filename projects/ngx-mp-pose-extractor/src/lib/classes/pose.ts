import { POSE_LANDMARKS, Results } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
import { PoseItem } from '../interfaces/pose-item';
import { PoseJson } from '../interfaces/pose-json';
import { PoseJsonItem } from '../interfaces/pose-json-item';
import { PoseVector } from '../interfaces/pose-vector';

// @ts-ignore
import cosSimilarity from 'cos-similarity';
import { SimilarPoseItem } from '../interfaces/matched-pose-item';

export class Pose {
  public generator?: string;
  public version?: number;
  public generatedVersion?: number;
  private videoMetadata!: {
    name: string;
    width: number;
    height: number;
    duration: number;
  };
  public poses: PoseItem[] = [];
  public isFinalized?: boolean = false;

  public static readonly IS_ENABLE_DUPLICATED_POSE_REDUCTION = true;

  public static readonly POSE_VECTOR_MAPPINGS = [
    // 右腕
    'rightWristToRightElbow',
    'rightElbowToRightShoulder',
    // 左腕
    'leftWristToLeftElbow',
    'leftElbowToLeftShoulder',
    // 右親指
    'rightThumbToWrist',
    // 左親指
    'leftThumbToWrist',
    // 右人差し指
    'rightIndexFingerToWrist',
    // 左人差し指
    'leftIndexFingerToWrist',
    // 右小指
    'rightPinkyFingerToWrist',
    // 左小指
    'leftPinkyFingerToWrist',
  ];

  public static readonly HAND_VECTOR_MAPPINGS = [
    // 右親指
    'rightThumbTipToFirstJoint',
    'rightThumbFirstJointToSecondsJoint',
    // 左親指
    'leftThumbTipToFirstJoint',
    'leftThumbFirstJointToSecondsJoint',
    // 右人差し指
    'rightIndexFingerTipToFirstJoint',
    'rightIndexFingerFirstJointToSecondsJoint',
    // 左人差し指
    'leftIndexFingerTipToFirstJoint',
    'leftIndexFingerFirstJointToSecondsJoint',
    // 右中指
    'rightMiddleFingerTipToFirstJoint',
    'rightMiddleFingerFirstJointToSecondsJoint',
    // 左中指
    'leftMiddleFingerTipToFirstJoint',
    'leftMiddleFingerFirstJointToSecondsJoint',
    // 右薬指
    'rightRingFingerTipToFirstJoint',
    'rightingFingerFirstJointToSecondsJoint',
    // 左薬指
    'leftRingFingerTipToFirstJoint',
    'leftRingFingerFirstJointToSecondsJoint',
    // 右小指
    'rightPinkyFingerTipToFirstJoint',
    'rightPinkyFingerFirstJointToSecondsJoint',
    // 左小指
    'leftPinkyFingerTipToFirstJoint',
    'leftPinkyFingerFirstJointToSecondsJoint',
  ];

  public static readonly POSE_JSON_VERSION = 1.1;

  constructor() {
    this.videoMetadata = {
      name: '',
      width: 0,
      height: 0,
      duration: 0,
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

  getPoses(): PoseItem[] {
    if (this.poses === undefined) return [];
    return this.poses;
  }

  getPoseByTime(timeMiliseconds: number): PoseItem | undefined {
    if (this.poses === undefined) return undefined;
    return this.poses.find((pose) => pose.timeMiliseconds === timeMiliseconds);
  }

  pushPose(
    videoTimeMiliseconds: number,
    frameImageJpegDataUrl: string | undefined,
    poseImageJpegDataUrl: string | undefined,
    videoWidth: number,
    videoHeight: number,
    videoDuration: number,
    results: Results
  ) {
    this.setVideoMetaData(videoWidth, videoHeight, videoDuration);

    if (results.poseLandmarks === undefined) return;

    const poseLandmarksWithWorldCoordinate: any[] = (results as any).ea
      ? (results as any).ea
      : [];
    if (poseLandmarksWithWorldCoordinate.length === 0) {
      console.warn(
        `[Pose] pushPose - Could not get the pose with the world coordinate`,
        results
      );
      return;
    }

    const poseVector = Pose.getPoseVector(poseLandmarksWithWorldCoordinate);
    if (!poseVector) {
      console.warn(
        `[Pose] pushPose - Could not get the pose vector`,
        poseLandmarksWithWorldCoordinate
      );
      return;
    }

    const pose: PoseItem = {
      timeMiliseconds: videoTimeMiliseconds,
      durationMiliseconds: -1,
      pose: poseLandmarksWithWorldCoordinate.map((landmark) => {
        return [landmark.x, landmark.y, landmark.z, landmark.visibility];
      }),
      vectors: poseVector,
      frameImageDataUrl: frameImageJpegDataUrl,
      poseImageDataUrl: poseImageJpegDataUrl,
    };

    if (1 <= this.poses.length) {
      const lastPose = this.poses[this.poses.length - 1];
      if (Pose.isSimilarPose(lastPose.vectors, pose.vectors)) {
        return;
      }

      // 前回のポーズの持続時間を設定
      const poseDurationMiliseconds =
        videoTimeMiliseconds - lastPose.timeMiliseconds;
      this.poses[this.poses.length - 1].durationMiliseconds =
        poseDurationMiliseconds;
    }

    this.poses.push(pose);
  }

  finalize() {
    if (0 == this.poses.length) {
      this.isFinalized = true;
      return;
    }

    // 全ポーズを比較して類似ポーズを削除
    if (Pose.IS_ENABLE_DUPLICATED_POSE_REDUCTION) {
      const newPoses: PoseItem[] = [];
      for (const poseA of this.poses) {
        let isDuplicated = false;
        for (const poseB of newPoses) {
          if (Pose.isSimilarPose(poseA.vectors, poseB.vectors)) {
            isDuplicated = true;
            break;
          }
        }
        if (isDuplicated) continue;

        newPoses.push(poseA);
      }

      console.info(
        `[Pose] getJson - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`
      );
      this.poses = newPoses;
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

    this.isFinalized = true;
  }

  getSimilarPoses(
    results: Results,
    bodyThreshold: number = 0.9,
    handThreshold: number | undefined = undefined
  ): SimilarPoseItem[] {
    const poseVector = Pose.getPoseVector((results as any).ea);
    if (!poseVector) throw 'Could not get the pose vector';

    const poses = [];
    for (const pose of this.poses) {
      let isSimilar = false;

      let bodySimilarity = Pose.getPoseBodySimilarity(pose.vectors, poseVector);
      if (bodyThreshold <= bodySimilarity) {
        isSimilar = true;
      }

      let handSimilarity;
      if (handThreshold !== undefined) {
        handSimilarity = Pose.getPoseHandSimilarity(pose.vectors, poseVector);
        if (handSimilarity !== undefined && handThreshold <= handSimilarity) {
          isSimilar = true;
        }
      }

      if (isSimilar) {
        poses.push({
          ...pose,
          similarity: bodySimilarity,
          bodySimilarity: bodySimilarity,
          handSimilarity: handSimilarity,
        });
      }
    }

    return poses;
  }

  static getPoseVector(
    poseLandmarks: { x: number; y: number; z: number }[]
  ): PoseVector | undefined {
    return {
      // 右腕
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
      // 左腕
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
      // 右親指
      rightThumbToWrist: [
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].x -
          poseLandmarks[POSE_LANDMARKS.RIGHT_THUMB].x,
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].y -
          poseLandmarks[POSE_LANDMARKS.RIGHT_THUMB].y,
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].z -
          poseLandmarks[POSE_LANDMARKS.RIGHT_THUMB].z,
      ],
      // 左親指
      leftThumbToWrist: [
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].x -
          poseLandmarks[POSE_LANDMARKS.LEFT_THUMB].x,
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].y -
          poseLandmarks[POSE_LANDMARKS.LEFT_THUMB].y,
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].z -
          poseLandmarks[POSE_LANDMARKS.LEFT_THUMB].z,
      ],
      // 右人差し指
      rightIndexFingerToWrist: [
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].x -
          poseLandmarks[POSE_LANDMARKS.RIGHT_INDEX].x,
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].y -
          poseLandmarks[POSE_LANDMARKS.RIGHT_INDEX].y,
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].z -
          poseLandmarks[POSE_LANDMARKS.RIGHT_INDEX].z,
      ],
      // 左人差し指
      leftIndexFingerToWrist: [
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].x -
          poseLandmarks[POSE_LANDMARKS.LEFT_INDEX].x,
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].y -
          poseLandmarks[POSE_LANDMARKS.LEFT_INDEX].y,
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].z -
          poseLandmarks[POSE_LANDMARKS.LEFT_INDEX].z,
      ],
      // 右小指
      rightPinkyFingerToWrist: [
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].x -
          poseLandmarks[POSE_LANDMARKS.RIGHT_PINKY].x,
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].y -
          poseLandmarks[POSE_LANDMARKS.RIGHT_PINKY].y,
        poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].z -
          poseLandmarks[POSE_LANDMARKS.RIGHT_PINKY].z,
      ],
      // 左小指
      leftPinkyFingerToWrist: [
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].x -
          poseLandmarks[POSE_LANDMARKS.LEFT_PINKY].x,
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].y -
          poseLandmarks[POSE_LANDMARKS.LEFT_PINKY].y,
        poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].z -
          poseLandmarks[POSE_LANDMARKS.LEFT_PINKY].z,
      ],
    };
  }

  static getHandVector() {
    throw new Error(`[Pose] getHandVector is not implemented yet`);
  }

  static isSimilarPose(
    poseVectorA: PoseVector,
    poseVectorB: PoseVector,
    threshold = 0.9
  ): boolean {
    let isSimilar = false;

    const poseBodySimilarity = Pose.getPoseBodySimilarity(
      poseVectorA,
      poseVectorB
    );
    if (poseBodySimilarity >= threshold) isSimilar = true;
    return isSimilar;
  }

  static getPoseBodySimilarity(
    poseVectorA: PoseVector,
    poseVectorB: PoseVector
  ): number {
    const bodyCosSimilarities = {
      leftWristToLeftElbow: cosSimilarity(
        poseVectorA.leftWristToLeftElbow,
        poseVectorB.leftWristToLeftElbow
      ),
      leftElbowToLeftShoulder: cosSimilarity(
        poseVectorA.leftElbowToLeftShoulder,
        poseVectorB.leftElbowToLeftShoulder
      ),
      rightWristToRightElbow: cosSimilarity(
        poseVectorA.rightWristToRightElbow,
        poseVectorB.rightWristToRightElbow
      ),
      rightElbowToRightShoulder: cosSimilarity(
        poseVectorA.rightElbowToRightShoulder,
        poseVectorB.rightElbowToRightShoulder
      ),
    };

    const bodyCosSimilaritiesSum = Object.values(bodyCosSimilarities).reduce(
      (sum, value) => sum + value,
      0
    );

    return bodyCosSimilaritiesSum / Object.keys(bodyCosSimilarities).length;
  }

  static getPoseHandSimilarity(
    poseVectorA: PoseVector,
    poseVectorB: PoseVector
  ): number | undefined {
    if (poseVectorA.leftThumbToWrist === undefined) return undefined;

    const handCosSimilarities = {
      leftThumbToWrist: cosSimilarity(
        poseVectorA.leftThumbToWrist,
        poseVectorB.leftThumbToWrist
      ),
      rightThumbToWrist: cosSimilarity(
        poseVectorA.rightThumbToWrist,
        poseVectorB.rightThumbToWrist
      ),
      leftIndexFingerToWrist: cosSimilarity(
        poseVectorA.leftIndexFingerToWrist,
        poseVectorB.leftIndexFingerToWrist
      ),
      rightIndexFingerToWrist: cosSimilarity(
        poseVectorA.rightIndexFingerToWrist,
        poseVectorB.rightIndexFingerToWrist
      ),
      leftPinkyFingerToWrist: cosSimilarity(
        poseVectorA.leftPinkyFingerToWrist,
        poseVectorB.leftPinkyFingerToWrist
      ),
      rightPinkyFingerToWrist: cosSimilarity(
        poseVectorA.rightPinkyFingerToWrist,
        poseVectorB.rightPinkyFingerToWrist
      ),
    };

    const handCosSimilaritiesSum = Object.values(handCosSimilarities).reduce(
      (sum, value) => sum + value,
      0
    );

    let hand = handCosSimilaritiesSum / Object.keys(handCosSimilarities).length;
    return hand;
  }

  public async getZip(): Promise<Blob> {
    const jsZip = new JSZip();
    jsZip.file('poses.json', this.getJson());

    for (const pose of this.poses) {
      if (pose.frameImageDataUrl) {
        try {
          const index =
            pose.frameImageDataUrl.indexOf('base64,') + 'base64,'.length;
          const base64 = pose.frameImageDataUrl.substring(index);
          jsZip.file(`frame-${pose.timeMiliseconds}.jpg`, base64, {
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
          jsZip.file(`pose-${pose.timeMiliseconds}.jpg`, base64, {
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
    }

    return await jsZip.generateAsync({ type: 'blob' });
  }

  public getJson(): string {
    if (this.videoMetadata === undefined || this.poses === undefined)
      return '{}';

    if (!this.isFinalized) {
      this.finalize();
    }

    let poseLandmarkMappings = [];
    for (const key of Object.keys(POSE_LANDMARKS)) {
      const index: number = POSE_LANDMARKS[key as keyof typeof POSE_LANDMARKS];
      poseLandmarkMappings[index] = key;
    }

    const json: PoseJson = {
      generator: 'mp-video-pose-extractor',
      version:
        this.version !== undefined ? this.version : Pose.POSE_JSON_VERSION,
      generatedVersion:
        this.generatedVersion !== undefined
          ? this.generatedVersion
          : Pose.POSE_JSON_VERSION,
      video: this.videoMetadata!,
      poses: this.poses.map((pose: PoseItem): PoseJsonItem => {
        return {
          t: pose.timeMiliseconds,
          d: pose.durationMiliseconds,
          pose: pose.pose,
          vectors: this.convertPoseVectorForJson(pose.vectors),
        };
      }),
      poseLandmarkMapppings: poseLandmarkMappings,
    };

    return JSON.stringify(json);
  }

  loadJson(json: string | any) {
    let parsedJson = typeof json === 'string' ? JSON.parse(json) : json;

    if (parsedJson.generator !== 'mp-video-pose-extractor') {
      throw '不正なファイル';
    }

    // バージョン確認・アップグレード
    if (parsedJson.version < Pose.POSE_JSON_VERSION) {
      parsedJson = this.upgradeJsonVersion(parsedJson);
    } else if (2.0 <= parsedJson.version) {
      throw '非対応のバージョン';
    }

    this.version = parsedJson.version;
    this.generatedVersion = (
      parsedJson.generatedVersion
        ? parsedJson.generatedVersion
        : parsedJson.version
    ) as number;

    this.videoMetadata = parsedJson.video;
    let poses = parsedJson.poses.map((poseJsonItem: PoseJsonItem): PoseItem => {
      const poseVector: any = {};
      Pose.POSE_VECTOR_MAPPINGS.map((key, index) => {
        poseVector[key as keyof PoseVector] = poseJsonItem.vectors[index];
      });

      return {
        timeMiliseconds: poseJsonItem.t,
        durationMiliseconds: poseJsonItem.d,
        pose: poseJsonItem.pose,
        vectors: poseVector,
        frameImageDataUrl: undefined,
      };
    });

    this.poses = poses;
  }

  async loadZip(buffer: ArrayBuffer, includeImages: boolean = true) {
    console.log(`[Pose] loadZip...`, JSZip);
    const jsZip = new JSZip();
    console.log(`[Pose] init...`);
    const zip = await jsZip.loadAsync(buffer, { base64: false });
    if (!zip) throw 'ZIPファイルを読み込めませんでした';

    const json = await zip.file('poses.json')?.async('text');
    if (json === undefined) {
      throw 'ZIPファイルに pose.json が含まれていません';
    }

    this.loadJson(json);

    if (includeImages) {
      for (const pose of this.poses) {
        if (!pose.frameImageDataUrl) {
          const frameImageFileName = `frame-${pose.timeMiliseconds}.jpg`;
          const imageBase64 = await zip
            .file(frameImageFileName)
            ?.async('base64');
          if (imageBase64) {
            pose.frameImageDataUrl = `data:image/jpeg;base64,${imageBase64}`;
          }
        }
        if (!pose.poseImageDataUrl) {
          const poseImageFileName = `pose-${pose.timeMiliseconds}.jpg`;
          const imageBase64 = await zip
            .file(poseImageFileName)
            ?.async('base64');
          if (imageBase64) {
            pose.poseImageDataUrl = `data:image/jpeg;base64,${imageBase64}`;
          }
        }
      }
    }
  }

  private upgradeJsonVersion(parsedJson: any) {
    console.log(
      `[Pose] upgradeJsonVersion - v${parsedJson.version}  -> v${Pose.POSE_JSON_VERSION}`
    );

    const oldVersion = parsedJson.version;
    if (oldVersion === 1) {
      parsedJson.version = 1.1;
      parsedJson.poses = parsedJson.poses.map(
        (poseJsonItem: PoseJsonItem): any => {
          let vectors: number[][] | undefined;
          if (poseJsonItem.pose !== undefined) {
            let poseLandmarks: {
              x: number;
              y: number;
              z: number;
            }[] = [];
            for (const part of poseJsonItem.pose) {
              poseLandmarks.push({
                x: part[0],
                y: part[1],
                z: part[2],
              });
            }
            const newVectors = Pose.getPoseVector(poseLandmarks);
            if (newVectors !== undefined) {
              const newJsonVectors = this.convertPoseVectorForJson(newVectors);
              console.log(
                `[Pose] upgradeJsonVersion - Convert vectors`,
                poseJsonItem.vectors,
                newJsonVectors
              );
              vectors = newJsonVectors;
            } else {
              console.log(
                `[Pose] upgradeJsonVersion - Failed to convert vectors`,
                poseLandmarks
              );
            }
          }

          return {
            t: poseJsonItem.t,
            d: poseJsonItem.d,
            pose: poseJsonItem.pose,
            vectors: vectors !== undefined ? vectors : poseJsonItem.vectors,
          };
        }
      );
    }

    return parsedJson;
  }

  private convertPoseVectorForJson(poseVector: PoseVector) {
    const newPoseVector = [];
    for (const key of Pose.POSE_VECTOR_MAPPINGS) {
      newPoseVector.push(poseVector[key as keyof PoseVector]);
    }
    return newPoseVector;
  }
}
