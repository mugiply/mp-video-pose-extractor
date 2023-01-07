import { TestBed } from '@angular/core/testing';

import { NgxMpPoseExtractorService } from './ngx-mp-pose-extractor.service';

describe('NgxMpPoseExtractorService', () => {
  let service: NgxMpPoseExtractorService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(NgxMpPoseExtractorService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
