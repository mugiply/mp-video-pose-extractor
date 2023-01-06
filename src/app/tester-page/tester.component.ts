import {
  Component,
  ElementRef,
  Input,
  OnChanges,
  OnDestroy,
  OnInit,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Results } from '@mediapipe/holistic';
import { Subscription } from 'rxjs';
import { PoseItem } from '../shared/pose';
import { PoseExporterService } from '../shared/pose-exporter.service';
import { PoseExtractorService } from '../shared/pose-extractor.service';

@Component({
  selector: 'app-tester',
  templateUrl: './tester.component.html',
  styleUrls: ['../shared/shared.scss', './tester.component.scss'],
  providers: [PoseExtractorService],
})
export class TesterComponent implements OnInit, OnDestroy, OnChanges {
  @Input()
  public poseFileType?: 'zip' | 'json';

  @Input()
  public poseZipArrayBuffer?: ArrayBuffer;

  @Input()
  public poseJson?: string;

  @ViewChild('cameraVideo')
  public cameraVideoElement!: ElementRef;

  public cameraStream?: MediaStream;
  public cameraPosePreviewStream?: MediaStream;

  public similarPoses?: PoseItem[];

  public isPoseLoaded = false;

  private onResultsEventEmitterSubscription?: Subscription;

  constructor(
    private poseExtractorService: PoseExtractorService,
    private poseExporterService: PoseExporterService,
    private snackBar: MatSnackBar
  ) {}

  async ngOnInit() {
    this.onResultsEventEmitterSubscription =
      this.poseExtractorService.onResultsEventEmitter.subscribe(
        (results: { mpResults: Results; posePreviewImageDataUrl: string }) => {
          this.onPoseDetected(
            results.mpResults,
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

  async ngOnChanges(changes: SimpleChanges) {
    if (
      changes['poseFileType'] ||
      changes['poseZipArrayBuffer'] ||
      changes['poseJson']
    ) {
      this.loadPoses();
    }
  }

  private async loadPoses() {
    if (!this.poseFileType || (!this.poseZipArrayBuffer && !this.poseJson)) {
      return;
    }

    const message = this.snackBar.open('ポーズファイルを読み込んでいます...');

    try {
      if (this.poseFileType === 'zip' && this.poseZipArrayBuffer) {
        await this.poseExporterService.loadZip(this.poseZipArrayBuffer);
      } else if (this.poseJson) {
        this.poseExporterService.loadJson(this.poseJson);
      } else {
        throw 'Invalid poseFileType';
      }
    } catch (e: any) {
      message.dismiss();
      this.snackBar.open(
        'エラー: ポーズファイルの読み込みに失敗しました...' + e.toString()
      );
      return;
    }

    message.dismiss();
    this.snackBar.open(
      `${this.poseExporterService.getNumberOfPoses()} 件のポーズを読み込みました`
    );

    if (!this.isPoseLoaded) {
      this.isPoseLoaded = true;
      this.initCamera();
    }
  }

  private async initCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return;
    }

    this.cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 720,
        height: 1280,
      },
      audio: false,
    });

    this.onCameraFrame();

    this.cameraPosePreviewStream =
      this.poseExtractorService.getPosePreviewMediaStream();
  }

  private async onCameraFrame() {
    const videoElement = this.cameraVideoElement?.nativeElement;
    if (!videoElement) return;

    if (videoElement.paused || videoElement.ended) {
      setTimeout(() => {
        this.onCameraFrame();
      }, 500);
      return;
    }

    await this.poseExtractorService.onVideoFrame(videoElement);
    await new Promise(requestAnimationFrame);
    this.onCameraFrame();
  }

  private async onPoseDetected(
    mpResults: Results,
    posePreviewImageDataUrl: string
  ) {
    this.similarPoses = this.poseExporterService.getSimilarPoses(mpResults);
  }
}
