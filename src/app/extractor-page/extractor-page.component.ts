import {
  Component,
  ElementRef,
  NgZone,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { Results } from '@mediapipe/holistic';
import { PoseComposerService } from 'projects/ngx-mp-pose-extractor/src/lib/services/pose-composer.service';
import {
  PoseSet,
  PoseExtractorService,
} from 'projects/ngx-mp-pose-extractor/src/public-api';
import { Subscription } from 'rxjs';

// @ts-ignore
import { MP4Demuxer } from './mp4-decorder/demuxer_mp4.mjs';

@Component({
  selector: 'app-extractor-page',
  templateUrl: './extractor-page.component.html',
  styleUrls: ['../shared/shared.scss', './extractor-page.component.scss'],
  providers: [PoseExtractorService, PoseComposerService],
})
export class ExtractorPageComponent implements OnInit, OnDestroy {
  @ViewChild('sourceVideo')
  public sourceVideoElement?: ElementRef;

  //public sourceVideoUrl?: SafeResourceUrl;
  public sourceVideoStream?: MediaStream;
  public sourceVideoFileName?: string = undefined;

  public posePreviewMediaStream?: MediaStream;
  public handPreviewMediaStream?: MediaStream;

  public state: 'initial' | 'processing' | 'completed' = 'initial';

  public mathFloor = Math.floor;

  public poseSet?: PoseSet;

  public mp4boxFile: any;
  public sourceVideoFrames?: any[];
  private sourceVideoLoadTimer: any = null;
  private sourceFrameCanvas?: HTMLCanvasElement;
  private sourceFrameCanvasContext?: CanvasRenderingContext2D;
  private pendingFrame: any = null;
  private lastFrameDrawedAt?: number;

  private onResultsEventEmitterSubscription!: Subscription;

  constructor(
    public poseComposerService: PoseComposerService,
    private poseExtractorService: PoseExtractorService,
    private domSanitizer: DomSanitizer,
    private snackBar: MatSnackBar,
    private ngZone: NgZone
  ) {}

  ngOnInit(): void {
    this.onResultsEventEmitterSubscription =
      this.poseExtractorService.onResultsEventEmitter.subscribe(
        (results: {
          mpResults: Results;
          sourceImageDataUrl: string;
          posePreviewImageDataUrl: string;
        }) => {
          this.onPoseDetected(
            results.mpResults,
            results.sourceImageDataUrl,
            results.posePreviewImageDataUrl
          );
        }
      );
  }

  ngOnDestroy(): void {
    if (this.onResultsEventEmitterSubscription) {
      this.onResultsEventEmitterSubscription.unsubscribe();
    }
  }

  async onSourceVideoEnded(event: any) {
    if (!this.poseSet) return;

    let message = this.snackBar.open('最終処理をしています...');
    await this.poseSet.finalize();
    message.dismiss();

    this.state = 'completed';
    message = this.snackBar.open('検出が完了しました', '保存');
    message.onAction().subscribe(() => {
      this.downloadPosesAsZip();
    });
  }

  async onChooseSourceVideoFile(event: any) {
    const files: File[] = event.target.files;
    if (files.length === 0) return;

    const videoFile = files[0];
    const videoFileUrl = URL.createObjectURL(videoFile);
    //this.sourceVideoUrl = this.domSanitizer.bypassSecurityTrustResourceUrl(videoFileUrl);

    this.sourceVideoFileName = videoFile.name;
    const videoName = videoFile.name.split('.').slice(0, -1).join('.');

    this.state = 'processing';
    this.poseSet = this.poseComposerService.init(videoName);

    // 動画のデコードを開始
    this.sourceFrameCanvas = document.createElement('canvas');
    this.sourceFrameCanvasContext = this.sourceFrameCanvas.getContext('2d')!;
    this.sourceVideoFrames = [];

    // @ts-ignore
    const decoder = new VideoDecoder({
      output: (frame: any) => {
        // 動画のフレームを取得したとき
        this.onVideoFrame(frame);
        this.lastFrameDrawedAt = Date.now();
      },
      error: (e: any) => {
        console.error(
          `[ExtractorPageComponent] onChooseSourceVideoFile - Error occurred`,
          e
        );
      },
    });

    const demuxer = new MP4Demuxer(videoFileUrl, {
      onConfig: (config: any) => {
        decoder.configure(config);
      },
      onChunk: (chunk: any) => {
        decoder.decode(chunk);
      },
      setStatus: (type: string, message: string) => {
        console.log(
          `[ExtractorPageComponent] - MP4Demuxer setStatus`,
          type,
          message
        );
      },
    });

    // 動画の全フレームが揃うまで待つ
    this.lastFrameDrawedAt = Date.now();
    this.sourceVideoLoadTimer = setInterval(() => {
      if (Date.now() - this.lastFrameDrawedAt! > 1000) {
        clearInterval(this.sourceVideoLoadTimer);
        this.onVideoAllFramesLoaded();
      }
    }, 1000);
  }

  async onVideoFrame(frame: any) {
    console.log(`[ExtractorPageComponent] - onVideoFrame`, frame);

    this.sourceVideoFrames?.push(frame);

    if (this.pendingFrame) {
      this.pendingFrame.close();
    } else {
      requestAnimationFrame(() => {
        this.renderAnimationFrame();
      });
    }
    this.pendingFrame = frame;
  }

  async onVideoAllFramesLoaded() {
    const message = this.snackBar.open(
      this.sourceVideoFrames?.length +
        'frames からポーズ検出をおこなっています... '
    );

    this.posePreviewMediaStream =
      this.poseExtractorService.getPosePreviewMediaStream();

    this.handPreviewMediaStream =
      this.poseExtractorService.getHandPreviewMediaStream();

    // TODO: 指定フレームごとにポーズ検出処理をする
    // this.sourceVideoFrames[0].timestamp
  }

  async renderAnimationFrame() {
    if (
      !this.sourceFrameCanvas ||
      !this.sourceFrameCanvasContext ||
      !this.pendingFrame
    )
      return;

    this.sourceFrameCanvas.width = this.pendingFrame.displayWidth;
    this.sourceFrameCanvas.height = this.pendingFrame.displayHeight;
    this.sourceFrameCanvasContext.drawImage(
      this.pendingFrame,
      0,
      0,
      this.sourceFrameCanvas.width,
      this.sourceFrameCanvas.height
    );
    this.pendingFrame.close();

    this.ngZone.run(() => {
      if (!this.sourceVideoStream && this.sourceFrameCanvas) {
        this.sourceVideoStream = this.sourceFrameCanvas.captureStream(30);
      }
    });

    this.pendingFrame = null;
  }

  async onPoseDetected(
    results: Results,
    sourceImageDataUrl: string,
    posePreviewImageDataUrl: string
  ) {
    if (!this.poseSet) return;

    const videoElement = this.sourceVideoElement?.nativeElement;
    if (!videoElement) return;

    const sourceVideoTimeMiliseconds = Math.floor(
      this.sourceVideoElement?.nativeElement.currentTime * 1000
    );

    const sourceVideoDurationMiliseconds = Math.floor(
      this.sourceVideoElement?.nativeElement.duration * 1000
    );

    this.poseSet.pushPose(
      sourceVideoTimeMiliseconds,
      sourceImageDataUrl,
      posePreviewImageDataUrl,
      videoElement.videoWidth,
      videoElement.videoHeight,
      sourceVideoDurationMiliseconds,
      results
    );
  }

  public downloadPosesAsZip() {
    if (!this.poseSet) return;

    const message = this.snackBar.open('保存するデータを生成しています...');
    try {
      this.poseComposerService.downloadAsZip(this.poseSet);
    } catch (e: any) {
      console.error(e);
      message.dismiss();
      this.snackBar.open('エラー: ' + e.toString(), 'OK');
      return;
    }

    message.dismiss();
  }

  public downloadPosesAsJson() {
    if (!this.poseSet) return;

    const message = this.snackBar.open('保存するデータを生成しています...');
    try {
      this.poseComposerService.downloadAsJson(this.poseSet);
    } catch (e: any) {
      console.error(e);
      message.dismiss();
      this.snackBar.open('エラー: ' + e.toString(), 'OK');
      return;
    }

    message.dismiss();
  }
}
