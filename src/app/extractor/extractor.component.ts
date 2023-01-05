import { Component, ElementRef, ViewChild } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { ExtractorService } from './extractor.service';

@Component({
  selector: 'app-extractor',
  templateUrl: './extractor.component.html',
  styleUrls: ['./extractor.component.scss'],
})
export class ExtractorComponent {
  @ViewChild('sourceVideo')
  public sourceVideoElement?: ElementRef;

  public sourceVideoUrl: any;

  public posePreviewMediaStream?: MediaStream;
  public handPreviewMediaStream?: MediaStream;

  constructor(
    private domSanitizer: DomSanitizer,
    private extractorService: ExtractorService
  ) {}

  async onChooseSourceVideoFile(event: any) {
    const files: File[] = event.target.files;
    if (files.length === 0) return;

    const videoFile = files[0];
    const videoFileUrl = URL.createObjectURL(videoFile);
    this.sourceVideoUrl =
      this.domSanitizer.bypassSecurityTrustResourceUrl(videoFileUrl);

    await this.onVideoFrame();

    this.posePreviewMediaStream =
      this.extractorService.getPosePreviewMediaStream();

    this.handPreviewMediaStream =
      this.extractorService.getHandPreviewMediaStream();
  }

  async onVideoFrame() {
    const videoElement = this.sourceVideoElement?.nativeElement;
    if (!videoElement) return;

    if (videoElement.paused || videoElement.ended) {
      setTimeout(() => {
        this.onVideoFrame();
      }, 500);
      return;
    }

    await this.extractorService.onVideoFrame(videoElement);
    await new Promise(requestAnimationFrame);
    this.onVideoFrame();
  }
}
