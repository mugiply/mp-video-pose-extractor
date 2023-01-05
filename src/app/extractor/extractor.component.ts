import { Component } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-extractor',
  templateUrl: './extractor.component.html',
  styleUrls: ['./extractor.component.scss'],
})
export class ExtractorComponent {
  public sourceVideoUrl: any;

  constructor(private domSanitizer: DomSanitizer) {}

  async onChooseSourceVideoFile(event: any) {
    const files: File[] = event.target.files;
    if (files.length === 0) return;

    const videoFile = files[0];
    const videoFileUrl = URL.createObjectURL(videoFile);
    this.sourceVideoUrl =
      this.domSanitizer.bypassSecurityTrustResourceUrl(videoFileUrl);
  }
}
