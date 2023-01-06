import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExtractorPageComponent } from './extractor-page.component';

describe('ExtractorComponent', () => {
  let component: ExtractorPageComponent;
  let fixture: ComponentFixture<ExtractorPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ExtractorPageComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(ExtractorPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
