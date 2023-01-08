import { POSE_LANDMARKS, Results } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
import { PoseItem } from '../interfaces/pose-item';
import { PoseJson } from '../interfaces/pose-json';
import { PoseJsonItem } from '../interfaces/pose-json-item';
import { PoseVector } from '../interfaces/pose-vector';

// @ts-ignore
import cosSimilarity from 'cos-similarity';

export class Pose {
  public generator?: string;
  public version?: number;
  private videoMetadata!: {
    name: string;
    width: number;
    height: number;
    duration: number;
  };
  public poses: PoseItem[] = [];
  public isFinalized?: boolean = false;

  public static readonly IS_ENABLE_DUPLICATED_POSE_REDUCTION = true;

  public static readonly IS_SHRINK_RAW_POSE_DATA = false;

  public static readonly POSE_VECTOR_MAPPINGS = [
    'rightWristToRightElbow',
    'rightElbowToRightShoulder',
    'leftWristToLeftElbow',
    'leftElbowToLeftShoulder',
  ];

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

  pushPose(
    videoTimeMiliseconds: number,
    frameImageJpegDataUrl: string | undefined,
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
      t: videoTimeMiliseconds,
      pose: poseLandmarksWithWorldCoordinate.map((landmark) => {
        return [landmark.x, landmark.y, landmark.z, landmark.visibility];
      }),
      vectors: poseVector,
      frameImageDataUrl: frameImageJpegDataUrl,
    };

    if (1 <= this.poses.length) {
      const lastPose = this.poses[this.poses.length - 1];
      if (Pose.isSimilarPose(lastPose.vectors, pose.vectors)) {
        return;
      }
    }

    this.poses.push(pose);
  }

  finalize() {
    if (Pose.IS_ENABLE_DUPLICATED_POSE_REDUCTION) {
      // 全ポーズを走査して、類似するポーズを削除する
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

    this.isFinalized = true;
  }

  getSimilarPoses(results: Results): PoseItem[] {
    const poseVector = Pose.getPoseVector((results as any).ea);
    if (!poseVector) throw 'Could not get the pose vector';

    return this.poses.filter((p) => Pose.isSimilarPose(p.vectors, poseVector));
  }

  static getPoseVector(
    poseLandmarks: { x: number; y: number; z: number }[]
  ): PoseVector | undefined {
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

  static isSimilarPose(
    poseVectorA: PoseVector,
    poseVectorB: PoseVector,
    threshold = 0.9
  ): boolean {
    const cosSimilarities = {
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

    let isSimilar = false;
    const cosSimilaritiesSum = Object.values(cosSimilarities).reduce(
      (sum, value) => sum + value,
      0
    );
    if (cosSimilaritiesSum >= threshold * Object.keys(cosSimilarities).length)
      isSimilar = true;

    console.log(`[Pose] isSimilarPose`, isSimilar, cosSimilarities);

    return isSimilar;
  }

  public async getZip(): Promise<Blob> {
    const jsZip = new JSZip();
    jsZip.file('poses.json', this.getJson());

    for (const pose of this.poses) {
      if (!pose.frameImageDataUrl) continue;
      try {
        const index =
          pose.frameImageDataUrl.indexOf('base64,') + 'base64,'.length;
        const base64 = pose.frameImageDataUrl.substring(index);

        jsZip.file(`snapshot-${pose.t}.jpg`, base64, { base64: true });
      } catch (error) {
        console.warn(
          `[PoseExporterService] push - Could not push frame image`,
          error
        );
        throw error;
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
      version: 1,
      video: this.videoMetadata!,
      poses: this.poses.map((pose: PoseItem): PoseJsonItem => {
        const poseVector = [];
        for (const key of Pose.POSE_VECTOR_MAPPINGS) {
          poseVector.push(pose.vectors[key as keyof PoseVector]);
        }

        return {
          t: pose.t,
          pose: Pose.IS_SHRINK_RAW_POSE_DATA ? [] : pose.pose,
          vectors: poseVector,
        };
      }),
      poseLandmarkMapppings: Pose.IS_SHRINK_RAW_POSE_DATA
        ? []
        : poseLandmarkMappings,
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
    this.poses = parsedJson.poses.map(
      (poseJsonItem: PoseJsonItem): PoseItem => {
        const poseVector: any = {};
        Pose.POSE_VECTOR_MAPPINGS.map((key, index) => {
          poseVector[key as keyof PoseVector] = poseJsonItem.vectors[index];
        });

        return {
          t: poseJsonItem.t,
          pose: poseJsonItem.pose,
          vectors: poseVector,
          frameImageDataUrl: undefined,
        };
      }
    );
  }

  async loadZip(buffer: ArrayBuffer, includeImages: boolean = true) {
    const jsZip = new JSZip();
    const zip = await jsZip.loadAsync(buffer, { base64: false });
    if (!zip) throw 'ZIPファイルを読み込めませんでした';

    const json = await zip.file('poses.json')?.async('text');
    if (json === undefined) {
      throw 'ZIPファイルに pose.json が含まれていません';
    }

    this.loadJson(json);

    if (includeImages) {
      for (const pose of this.poses) {
        const frameImageFileName = `snapshot-${pose.t}.jpg`;
        const imageBase64 = await zip.file(frameImageFileName)?.async('base64');
        if (imageBase64 === undefined && !pose.frameImageDataUrl) {
          continue;
        }

        pose.frameImageDataUrl = `data:image/jpeg;base64,${imageBase64}`;
      }
    }
  }
}
