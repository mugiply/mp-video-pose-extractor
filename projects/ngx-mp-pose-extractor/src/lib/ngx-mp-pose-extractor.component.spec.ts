import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NgxMpPoseExtractorComponent } from './ngx-mp-pose-extractor.component';

describe('NgxMpPoseExtractorComponent', () => {
  let component: NgxMpPoseExtractorComponent;
  let fixture: ComponentFixture<NgxMpPoseExtractorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ NgxMpPoseExtractorComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(NgxMpPoseExtractorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
